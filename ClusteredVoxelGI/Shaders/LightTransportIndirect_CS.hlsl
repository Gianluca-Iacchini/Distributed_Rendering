#include "VoxelUtils.hlsli"

ConstantBuffer<ConstantBufferVoxelCommons> cbVoxelCommons : register(b0);
ConstantBuffer<ConstantBufferIndirectLightTransport> cbFrustumCulling : register(b1);

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space0);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space0);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space0);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space0);

StructuredBuffer<uint2> gVoxelFaceDataBuffer : register(t0, space1);
// The element i contains the start index in gVoxelFaceDataBuffer and the number of the faces for the voxel with index i
StructuredBuffer<uint2> gVoxelFaceStartCountBuffer : register(t1, space1);

StructuredBuffer<AABB> gVoxelAABBBuffer : register(t0, space2);
StructuredBuffer<ClusterAABBInfo> gClusterAABBInfoBuffer : register(t1, space2);
// Map from aabbVoxelIndices to gVoxelIndicesCompactBuffer.
StructuredBuffer<uint> gAABBVoxelIndices : register(t2, space2);

StructuredBuffer<uint2> gFaceClusterVisibility : register(t0, space3);
// Stores all the visible clusters for all the faces. Clusters visible from the same faced are stored in sequence.
StructuredBuffer<uint> gVisibleClustersBuffer : register(t1, space3);

ByteAddressBuffer gLitVoxels : register(t0, space4);
ByteAddressBuffer gLitClusters : register(t1, space4);

ByteAddressBuffer gVisibleVoxels : register(t0, space5);
ByteAddressBuffer gVisibleVoxelCounter : register(t1, space5);
StructuredBuffer<uint> gVisibleVoxelIndices : register(t2, space5);

RWStructuredBuffer<float> gVoxelRadiance : register(u0);


[numthreads(128, 1, 1)]
void CS( uint3 DTid : SV_DispatchThreadID )
{
    uint threadID = DTid.x;
    
    uint nVisibleVoxels = gVisibleVoxelCounter.Load(0);
    
    if (threadID > nVisibleVoxels)
        return;
    
    uint voxelIndex = gVisibleVoxelIndices[threadID];
    
    uint faceDir[6];
    // X field is the start index in gVoxelFaceDataBuffer, Y field is number of elements
    uint2 faceStartCount = gVoxelFaceStartCountBuffer[voxelIndex];
    
    float radiance = 0.0f;
    
    for (uint i = 0; i < faceStartCount.y; i++)
    {
        uint faceIdx = faceStartCount.x + i;
        faceDir[i] = gVoxelFaceDataBuffer[faceStartCount.x + i].y;
        
        uint2 clusterStartCount = gFaceClusterVisibility[faceIdx];

        for (uint clusterIdx = clusterStartCount.x; clusterIdx < clusterStartCount.x + clusterStartCount.y; clusterIdx++)
        {
            if (IsVoxelPresent(gVisibleClustersBuffer[clusterIdx], gLitClusters))
            {
                radiance += 0.005f;
            }
        }

    }
    
    gVoxelRadiance[voxelIndex] = radiance;
    
    //gVoxelRadiance[voxelIndex] = 0.5f;
}