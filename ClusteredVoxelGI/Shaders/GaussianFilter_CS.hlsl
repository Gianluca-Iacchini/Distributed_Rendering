#include "VoxelUtils.hlsli"

ConstantBuffer<ConstantBufferVoxelCommons> cbVoxelCommons : register(b0);
ConstantBuffer<ConstantBufferGaussianFilter> cbGaussianFilter : register(b1);

ByteAddressBuffer gVoxelOccupiedBuffer : register(t0, space0);

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space1);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space1);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space1);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space1);


StructuredBuffer<float> gFaceClusterPenaltyBuffer : register(t0, space2);
StructuredBuffer<float> gFaceCloseVoxelsPenaltyBuffer : register(t1, space2);

StructuredBuffer<uint2> gFaceRadianceReadBuffer : register(t0, space4);

StructuredBuffer<uint2> gReadFinalRadianceBuffer : register(t0, space5);

RWStructuredBuffer<uint2> gGaussianFirstFilterBuffer : register(u0, space0);
RWStructuredBuffer<float> gGaussianPrecomputedDataBuffer : register(u1, space0);

RWStructuredBuffer<uint2> gWriteFinalRadianceBuffer : register(u0, space1);

RWByteAddressBuffer gVisibleFacesCounter : register(u0, space2);
RWStructuredBuffer<uint> gGaussianFaceIndices : register(u2, space2);
RWByteAddressBuffer gIndirectLightUpdatedVoxelsBitmap : register(u3, space2);
RWByteAddressBuffer gGaussianUpdatedVoxelsBitmap : register(u4, space2);

#define SIDE 2
#define KERNEL_SIZE 2 * SIDE + 1
#define SIGMA 25.0

uint2 FindHashedCompactedPositionIndex(uint3 coord, uint3 gridDimension)
{
    uint2 result = uint2(0, 0); // y field is control value, 0 means element not found, 1 means element found
    uint indirectionIndex = gridDimension.z * coord.z + coord.y;
    uint index = gIndirectionIndexBuffer[indirectionIndex];
    uint rank = gIndirectionRankBuffer[indirectionIndex];
    uint hashedPosition = GetLinearCoord(coord, gridDimension);
    
    if (rank == 0)
        return result;
    
    uint tempHashed;
    uint startIndex = index;
    uint endIndex = index + rank;
    uint currentIndex = (startIndex + endIndex) / 2;

    for (int i = 0; i < int(12); ++i)
    {
        tempHashed = gVoxelHashedCompactBuffer[currentIndex];

        if (tempHashed == hashedPosition)
        {
            return uint2(currentIndex, 1);
        }

        if (tempHashed < hashedPosition)
        {
            startIndex = currentIndex;
            currentIndex = (startIndex + endIndex) / 2;
        }
        else
        {
            endIndex = currentIndex;
            currentIndex = (startIndex + endIndex) / 2;
        }
    }

    return result;
}

float gaussianDistribution(float x, float y, float z, float sigma)
{
    float denominator = 2.0 * PI * sigma * sigma;
    float expNumerator = x * x + y * y + z * z;
    float expDenominator = 2.0 * sigma * sigma;

    return (1.0 / denominator) * exp(-1.0 * (expNumerator / expDenominator));
}


float3 filterFace(uint voxelIdx, uint faceIdx, bool isFirstPass, out bool shouldSet)
{
    uint3 voxelTexCoords = GetVoxelPosition(gVoxelHashedCompactBuffer[voxelIdx], cbVoxelCommons.voxelTextureDimensions);

    shouldSet = true;
    float sum = 0.0f; // Normalization sum
    float3 filteredIrradiance = float3(0.0f, 0.0f, 0.0f);

    // Iterate over the kernel
    for (int x = -SIDE; x <= SIDE; ++x)
    {
        for (int y = -SIDE; y <= SIDE; ++y)
        {
            for (int z = -SIDE; z <= SIDE; ++z)
            {
                int3 offset = int3(x, y, z);
                if (IsWithinBounds(voxelTexCoords, offset, cbVoxelCommons.voxelTextureDimensions))
                {
                    uint3 neighbourCoord = voxelTexCoords + uint3(offset);
                    if (IsVoxelPresent(neighbourCoord, cbVoxelCommons.voxelTextureDimensions, gVoxelOccupiedBuffer))
                    {
                        uint neighbourIdx = FindHashedCompactedPositionIndex(neighbourCoord, cbVoxelCommons.voxelTextureDimensions).x;
                        float3 voxelFaceIrradiance = float3(0.0f, 0.0f, 0.0f);
                        
                        uint2 packedRadiance = uint2(0, 0);
                        if (cbGaussianFilter.CurrentPhase == 1)
                        {
                            packedRadiance = gFaceRadianceReadBuffer[neighbourIdx * 6 + faceIdx];
                            if (!IsVoxelPresent(neighbourIdx, gIndirectLightUpdatedVoxelsBitmap))
                            {
                                shouldSet = false;
                            }
                        }
                        else if (cbGaussianFilter.CurrentPhase == 2)
                        {
                            packedRadiance = gGaussianFirstFilterBuffer[neighbourIdx * 6 + faceIdx];
                        }
                        voxelFaceIrradiance.xy = UnpackFloats16(packedRadiance.x);
                        voxelFaceIrradiance.z = UnpackFloats16(packedRadiance.y).x;

                        if (voxelIdx == neighbourIdx)
                        {
                            voxelFaceIrradiance *= gFaceCloseVoxelsPenaltyBuffer[neighbourIdx * 6 + faceIdx];
                        }

                        if (isFirstPass || (any(voxelFaceIrradiance > 0.0f)))
                        {
                            uint linearCoord = GetLinearCoord(uint3(x + SIDE, y + SIDE, z + SIDE), uint3(KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE));
                            float gaussianValue = gGaussianPrecomputedDataBuffer[linearCoord];
                            sum += gaussianValue;

                            // Accumulate the filtered irradiance directly
                            filteredIrradiance += voxelFaceIrradiance * gaussianValue;
                        }
                    }
                }
            }
        }
    }

    // Normalize the result
    if (sum > 0.0f)
    {
        filteredIrradiance /= sum;
    }

    return filteredIrradiance;
}


[numthreads(128, 1, 1)]
void CS( uint3 DTid : SV_DispatchThreadID)
{
    uint threadGlobalIndex = DTid.x;
    
    if (cbGaussianFilter.CurrentPhase == 0)
    {
        if (threadGlobalIndex != 0)
            return;
        
        for (uint x = 0; x < KERNEL_SIZE; x++)
        {
            for (uint y = 0; y < KERNEL_SIZE; y++)
            {
                for (uint z = 0; z < KERNEL_SIZE; z++)
                {
                    int3 values = int3(int(x) - SIDE, int(y) - SIDE, int(z) - SIDE);
                    uint linearCoord = GetLinearCoord(uint3(x, y, z), uint3(KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE));
                    gGaussianPrecomputedDataBuffer[linearCoord] = gaussianDistribution(values.x, values.y, values.z, SIGMA);
                }
            }
        }
    }

    
    uint visibleFaces = gVisibleFacesCounter.Load(4);
    
    if (threadGlobalIndex >= visibleFaces)
        return;
    
    uint idx = gGaussianFaceIndices[threadGlobalIndex];
    
    uint voxIdx = (uint) floor(idx / 6.0f);
    uint faceIndex = idx % 6;
    
    if (cbGaussianFilter.CurrentPhase == 1)
    {
        uint2 packedRadiance = gFaceRadianceReadBuffer[idx];
        float3 radiance = float3(0.0f, 0.0f, 0.0f);
    
        radiance.xy = UnpackFloats16(packedRadiance.x);
        radiance.z = UnpackFloats16(packedRadiance.y).x;
    
        bool shouldSet = true;
        
        float3 filteredRadiance = filterFace(voxIdx, faceIndex, true, shouldSet);
    
        if (shouldSet)
        {
            SetVoxelPresence(voxIdx, gGaussianUpdatedVoxelsBitmap);
        }
        
        uint2 packedData = uint2(PackFloats16(filteredRadiance.xy), PackFloats16(float2(filteredRadiance.z, 0.0f)));
        gGaussianFirstFilterBuffer[idx] = packedData;
    }
    else if (cbGaussianFilter.CurrentPhase == 2)
    {
        uint2 packedRadiance = gGaussianFirstFilterBuffer[idx];
        float3 radiance = float3(0.0f, 0.0f, 0.0f);
    
        radiance.xy = UnpackFloats16(packedRadiance.x);
        radiance.z = UnpackFloats16(packedRadiance.y).x;
        
        if (any(radiance > 0.0f))
        {
            bool shouldSet;
            radiance = filterFace(voxIdx, faceIndex, false, shouldSet);
            packedRadiance = uint2(PackFloats16(radiance.xy), PackFloats16(float2(radiance.z, 0.0f)));
        }
        else
        {
            packedRadiance = uint2(0, 0);
        }
        
        gWriteFinalRadianceBuffer[idx] = packedRadiance;
    }
}