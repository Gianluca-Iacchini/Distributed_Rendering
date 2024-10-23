#include "VoxelUtils.hlsli"

ConstantBuffer<ConstantBufferVoxelCommons> cbVoxelCommons : register(b0);
ConstantBuffer<ConstantBufferGaussianFilter> cbGaussianFilter : register(b1);

ByteAddressBuffer gVoxelOccupiedBuffer : register(t0, space0);

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space1);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space1);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space1);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space1);

StructuredBuffer<uint2> gVoxelFaceDataBuffer : register(t0, space2);
// The element i contains the start index in gVoxelFaceDataBuffer and the number of the faces for the voxel with index i
StructuredBuffer<uint2> gVoxelFaceStartCountBuffer : register(t1, space2);

StructuredBuffer<float> gFaceClusterPenaltyBuffer : register(t0, space3);
StructuredBuffer<float> gFaceCloseVoxelsPenaltyBuffer : register(t1, space3);

ByteAddressBuffer gVisibleFaceCounter : register(t0, space4);
StructuredBuffer<uint> gVisibleFaceIndices : register(t1, space4);

RWStructuredBuffer<uint2> gFaceRadianceBuffer : register(u0, space0);
RWStructuredBuffer<uint2> gFaceFilteredRadianceBuffer : register(u1, space0);

#define SIDE 2
#define KERNEL_SIZE 2 * SIDE + 1

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


float3 filterFace(int faceIndex, RWStructuredBuffer<uint2> gRadianceBuffer, bool isFirstPass)
{
    uint2 faceData = gVoxelFaceDataBuffer[faceIndex];
    uint3 voxelTexCoords = GetVoxelPosition(gVoxelHashedCompactBuffer[faceData.x], cbVoxelCommons.voxelTextureDimensions);
    

    float gaussianValue;

    float3 voxelFaceIrradiance;

    float lKernel[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE];
    float3 lVoxelRadiance[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE];

    float sigma = 25.0f;
    float sum = 0.0f; // used for normalization, one sum value for each rgb channel
    
    uint2 currentFaceData = uint2(UINT_MAX, 0);
    uint currentFaceIdx = UINT_MAX;
    
    // Generate 3x3 kernel 
    for (int x = -SIDE; x <= SIDE; ++x)
    {
        for (int y = -SIDE; y <= SIDE; ++y)
        {
            for (int z = -SIDE; z <= SIDE; ++z)
            {
                lKernel[x + SIDE][y + SIDE][z + SIDE] = 0.0f;
                lVoxelRadiance[x + SIDE][y + SIDE][z + SIDE] = float3(0.0f, 0.0f, 0.0f);
                int3 offset = int3(x, y, z);

                if (IsWithinBounds(voxelTexCoords, offset, cbVoxelCommons.voxelTextureDimensions))
                {
                    uint3 neighbourCoord = voxelTexCoords + uint3(offset);
                    if (IsVoxelPresent(neighbourCoord, cbVoxelCommons.voxelTextureDimensions, gVoxelOccupiedBuffer))
                    {
                        uint neighbourIdx = FindHashedCompactedPositionIndex(neighbourCoord, cbVoxelCommons.voxelTextureDimensions).x;
                        uint2 faceStartCount = gVoxelFaceStartCountBuffer[neighbourIdx];
                        
                        voxelFaceIrradiance = float3(0.0f, 0.0f, 0.0f);
                        for (uint f = faceStartCount.x; f < faceStartCount.x + faceStartCount.y; ++f)
                        {
                            currentFaceIdx = f;
                            currentFaceData = gVoxelFaceDataBuffer[f];
                            if (currentFaceData.y == faceData.y)
                            {
                                uint2 packedRadiance = gRadianceBuffer[f];
                                voxelFaceIrradiance.xy = UnpackFloats16(packedRadiance.x);
                                voxelFaceIrradiance.z = UnpackFloats16(packedRadiance.y).x;
                                break;
                            }
                        }

                        if (faceData.x == currentFaceData.x)
                        {
                            voxelFaceIrradiance *= gFaceCloseVoxelsPenaltyBuffer[currentFaceIdx];
                        }

                        if (isFirstPass || (any(voxelFaceIrradiance > 0.0f)))
                        {
                            gaussianValue = gaussianDistribution(x, y, z, sigma);
                            lKernel[x + SIDE][y + SIDE][z + SIDE] = gaussianValue;
                            lVoxelRadiance[x + SIDE][y + SIDE][z + SIDE] = voxelFaceIrradiance;
                            sum += gaussianValue;
                        }
                    }
                }
            }
        }
    }

    // Avoid division by zero
    if (sum == 0.0f)
    {
        sum = 1.0f;
    }

    // Apply kernel for the face
    float3 filteredIrradiance = float3(0.0f, 0.0f, 0.0f);
    float kernelNormalizedValue;
    for (uint i = 0; i < KERNEL_SIZE; ++i)
    {
        for (uint j = 0; j < KERNEL_SIZE; ++j)
        {
            for (uint k = 0; k < KERNEL_SIZE; ++k)
            {
                kernelNormalizedValue = lKernel[i][j][k] /= sum;
                filteredIrradiance += lVoxelRadiance[i][j][k] * kernelNormalizedValue;
            }
        }
    }

    return filteredIrradiance;
}


[numthreads(128, 1, 1)]
void CS( uint3 DTid : SV_DispatchThreadID )
{
    uint threadGlobalIndex = DTid.x;
    
    uint visibleFaces = gVisibleFaceCounter.Load(0);
    
    if (threadGlobalIndex >= visibleFaces)
        return;
    
    uint faceIdx = gVisibleFaceIndices[threadGlobalIndex];
    
    if (cbGaussianFilter.CurrentPhase == 0)
    {
        uint2 packedRadiance = gFaceRadianceBuffer[faceIdx];
        float3 radiance = float3(0.0f, 0.0f, 0.0f);
    
        radiance.xy = UnpackFloats16(packedRadiance.x);
        radiance.z = UnpackFloats16(packedRadiance.y).x;
    
        float3 filteredRadiance = filterFace(faceIdx, gFaceRadianceBuffer, true);
    
        uint2 packedData = uint2(PackFloats16(filteredRadiance.xy), PackFloats16(float2(filteredRadiance.z, 0.0f)));
    
        gFaceFilteredRadianceBuffer[faceIdx] = packedData;
    }
    else if (cbGaussianFilter.CurrentPhase == 1)
    {
        uint2 packedRadiance = gFaceFilteredRadianceBuffer[faceIdx];
        float3 radiance = float3(0.0f, 0.0f, 0.0f);
    
        radiance.xy = UnpackFloats16(packedRadiance.x);
        radiance.z = UnpackFloats16(packedRadiance.y).x;
        
        if (any(radiance > 0.0f))
        {
            radiance = filterFace(faceIdx, gFaceFilteredRadianceBuffer, false);
            packedRadiance = uint2(PackFloats16(radiance.xy), PackFloats16(float2(radiance.z, 0.0f)));
        }
        else
        {
            packedRadiance = uint2(0, 0);
        }
        
        gFaceRadianceBuffer[faceIdx] = packedRadiance;
    }
}