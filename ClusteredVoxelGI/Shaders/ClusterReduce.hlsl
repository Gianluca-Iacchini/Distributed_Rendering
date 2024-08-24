#include "VoxelUtils.hlsli"

cbuffer SLICCommon : register(b0)
{
    uint CurrentPhase;
    uint NumberOfSubclusters;
    uint NumberOfSuperClusters; // Number of clusters
    float m; // Compactness factor
    
    uint3 GridDimension;
    uint S; // Cluster area
    
    uint3 TileDimension;
    uint FirstClusterSet;
    
    uint3 TilesToUpdate;
    uint CurrentIteration;
}


// Buffer containing the super cluster data
RWStructuredBuffer<ClusterData> gSuperClusterDataBuffer : register(u0, space0);
// Buffer containing the linked list of voxel for each cluster
RWStructuredBuffer<uint> gNextVoxelBuffer : register(u1, space0);
// Buffers containing the voxel assignment and distance map
RWStructuredBuffer<uint> gVoxelAssignmentBuffer : register(u2, space0);
RWStructuredBuffer<float> gVoxelDistanceBuffer : register(u3, space0);

// Tiles for the super cluster
RWTexture3D<uint> gTileBuffer : register(u4, space0);
// Linked list of all the super clusters in a tile
RWStructuredBuffer<uint> gNextSuperClusterInTile : register(u5, space0);
// Generic counter buffer
RWStructuredBuffer<uint> gClusterCounterBuffer : register(u6, space0);

// Buffer containing the sub cluster data
RWStructuredBuffer<ClusterData> gSubClusterDataBuffer : register(u7, space0);
// Linked list of all the subclusters assigned to a super cluster
RWStructuredBuffer<uint> gNextSubClusterInSuperCluster : register(u8, space0);



[numthreads(8, 8, 8)]
void main( uint3 TileID : SV_DispatchThreadID, uint GroupIndex : SV_GroupIndex )
{
    // We reuse the original TileMap, and at every iteration we reduce the number of tiles by 8 (2 in every dimension).
    // So we need to offset the TileMap by 2 the first iteration, 4 the seconds, 8 the third, etc.
    // We start at 1 because the iteration 0 is the original one for voxelization
    uint tileMapOffset = pow(2, CurrentIteration + 1);
    
    TileID = TileID * tileMapOffset;
    
    // Merge tiles tiles, 8 to 1.
    if (CurrentPhase == 0)
    {

    }
    else if (CurrentPhase == 1)
    {
                // Flattened global thread id for a 1D dispatch;
        // 512 = 8 * 8 * 8
        uint threadLinearId = TileID.x * 512 + GroupIndex;
        
        if (threadLinearId >= NumberOfSuperClusters)
            return;
        
        ClusterData superClusterData = gSuperClusterDataBuffer[threadLinearId];
        
        uint subClusterIndex = superClusterData.FirstDataIndex;
        
        float3 avgPos = float3(0.0f, 0.0f, 0.0f);
        float3 avgNormal = float3(0.0f, 0.0f, 0.0f);
        uint subClusterCount = 0;
        
        while (subClusterCount != UINT_MAX)
        {
            ClusterData subClusterData = gSubClusterDataBuffer[subClusterIndex];
            avgPos += subClusterData.Center;
            avgNormal += subClusterData.Normal;
            subClusterCount++;
            
            
            // Get the next sub cluster
            subClusterIndex = gNextSubClusterInSuperCluster[subClusterIndex];
        }

        if (FirstClusterSet > 0 && subClusterCount > 0)
        {
            superClusterData.Center = avgPos / subClusterCount;
            superClusterData.Normal = normalize(avgNormal);
        }
        superClusterData.VoxelCount = subClusterCount;
        superClusterData.FirstDataIndex = UINT_MAX;
        
        gSuperClusterDataBuffer[threadLinearId] = superClusterData;
        
        uint3 tileCoord = (uint3) floor(superClusterData.Center / (2 * S));
            
        uint prev = UINT_MAX;
        uint currentValue;
        InterlockedCompareExchange(gTileBuffer[tileCoord], prev, threadLinearId, currentValue);
        
        [allow_uav_condition]
        while (currentValue != prev)
        {
            prev = currentValue;
            gNextSuperClusterInTile[threadLinearId] = currentValue;
            InterlockedCompareExchange(gTileBuffer[tileCoord], prev, threadLinearId, currentValue);
        }
    }
}