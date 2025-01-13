#include "../../VoxelUtils/Shaders/VoxelUtils.hlsli"

ConstantBuffer<ConstantBufferVoxelCommons> cbVoxelCommons : register(b0);
ConstantBuffer<ConstantBufferGaussianFilter> cbGaussianFilter : register(b1);

ByteAddressBuffer gVoxelOccupiedBuffer : register(t0, space0);

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space1);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space1);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space1);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space1);


StructuredBuffer<uint> gFaceRadianceReadBuffer : register(t0, space4);


RWStructuredBuffer<uint> gGaussianFirstFilterBuffer : register(u0, space0);
RWStructuredBuffer<float> gGaussianPrecomputedDataBuffer : register(u1, space0);

RWStructuredBuffer<uint2> gWriteFinalRadianceBuffer : register(u0, space1);

RWByteAddressBuffer gVisibleFacesCounter : register(u0, space2);
RWStructuredBuffer<uint> gGaussianFaceIndices : register(u2, space2);
RWByteAddressBuffer gIndirectLightUpdatedVoxelsBitmap : register(u3, space2);
RWByteAddressBuffer gGaussianUpdatedVoxelsBitmap : register(u4, space2);

#define DEFAULT_SIDE 2
#define DEFAULT_KERNEL_SIZE 2 * DEFAULT_SIDE + 1
#define SIGMA 2.0

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

    int kernelSide = (cbGaussianFilter.KernelSize - 1) / 2;
    
    // Iterate over the kernel
    for (int x = -kernelSide; x <= kernelSide; ++x)
    {
        for (int y = -kernelSide; y <= kernelSide; ++y)
        {
            for (int z = -kernelSide; z <= kernelSide; ++z)
            {
                int3 offset = int3(x, y, z);
                if (IsWithinBounds(voxelTexCoords, offset, cbVoxelCommons.voxelTextureDimensions))
                {
                    uint3 neighbourCoord = voxelTexCoords + uint3(offset);
                    if (IsVoxelPresent(neighbourCoord, cbVoxelCommons.voxelTextureDimensions, gVoxelOccupiedBuffer))
                    {
                        uint neighbourIdx = FindHashedCompactedPositionIndex(neighbourCoord, cbVoxelCommons.voxelTextureDimensions).x;
                        float3 voxelFaceIrradiance = float3(0.0f, 0.0f, 0.0f);
                        
                        uint packedRadiance = 0;
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
                        
                        voxelFaceIrradiance = UnpackUintToFloat3(packedRadiance);


                        if (isFirstPass || (any(voxelFaceIrradiance > 0.0f)))
                        {
                            float gaussianValue = 0.0f;
                            
                            if (cbGaussianFilter.UsePreComputedGaussian == 1)
                            {
                                uint linearCoord = GetLinearCoord(uint3(x + DEFAULT_SIDE, y + DEFAULT_SIDE, z + DEFAULT_SIDE), uint3(DEFAULT_KERNEL_SIZE, DEFAULT_KERNEL_SIZE, DEFAULT_KERNEL_SIZE));
                                gaussianValue = gGaussianPrecomputedDataBuffer[linearCoord];
                            }
                            else
                            {
                                gaussianValue = gaussianDistribution(float(x), float(y), float(z), cbGaussianFilter.Sigma);
                            }
                            
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
        
        for (uint x = 0; x < DEFAULT_KERNEL_SIZE; x++)
        {
            for (uint y = 0; y < DEFAULT_KERNEL_SIZE; y++)
            {
                for (uint z = 0; z < DEFAULT_KERNEL_SIZE; z++)
                {
                    int3 values = int3(int(x) - DEFAULT_SIDE, int(y) - DEFAULT_SIDE, int(z) - DEFAULT_SIDE);
                    uint linearCoord = GetLinearCoord(uint3(x, y, z), uint3(DEFAULT_KERNEL_SIZE, DEFAULT_KERNEL_SIZE, DEFAULT_KERNEL_SIZE));
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

        float3 radiance = float3(0.0f, 0.0f, 0.0f);

    
        if (cbGaussianFilter.PassCount > 0)
        {
            bool shouldSet = true;
        
            float3 filteredRadiance = filterFace(voxIdx, faceIndex, true, shouldSet);
    
            if (shouldSet)
            {
                SetVoxelPresence(voxIdx, gGaussianUpdatedVoxelsBitmap);
            }
            
            radiance = filteredRadiance;
        }
        else
        {
            uint packedRadiance = gFaceRadianceReadBuffer[idx];
            radiance.xyz = UnpackUintToFloat3(packedRadiance);
        }

        uint packedData = PackFloat3ToUint(radiance);
        gGaussianFirstFilterBuffer[idx] = packedData;
        
    }
    else if (cbGaussianFilter.CurrentPhase == 2)
    {

        uint2 finalPacked = uint2(0, 0);
        uint packedRadiance = gGaussianFirstFilterBuffer[idx];
        float3 radiance = UnpackUintToFloat3(packedRadiance);
        
        if (cbGaussianFilter.PassCount > 1)
        {

    
        

        
            if (any(radiance > 0.0f))
            {
                bool shouldSet;
                radiance = filterFace(voxIdx, faceIndex, false, shouldSet);
                finalPacked = uint2(PackFloats16(radiance.xy), PackFloats16(float2(radiance.z, 0.0f)));
            }
            else
            {
                finalPacked = uint2(0, 0);
            }
        }
        else
        {
            finalPacked = uint2(PackFloats16(radiance.xy), PackFloats16(float2(radiance.z, 0.0f)));
        }
        
        gWriteFinalRadianceBuffer[idx] = finalPacked;
    }
}