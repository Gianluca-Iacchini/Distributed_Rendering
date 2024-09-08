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
    
    uint CurrentIteration;
    uint VoxelCount;
    float _pad1;
    float _pad2;
}

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space0);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space0);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space0);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space0);

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

RWStructuredBuffer<uint> gSubClusterAssignmentBuffer : register(u7, space0);
// Linked list of all the subclusters assigned to a super cluster
RWStructuredBuffer<uint> gNextSubClusterInSuperCluster : register(u8, space0);

// Buffer containing the sub cluster data
RWStructuredBuffer<ClusterData> gSubClusterDataBuffer : register(u9, space0);


float3 SetClusterNormalDirection(float3 normal)
{
    int3 axisDirections[6] =
    {
        int3(1, 0, 0),
        int3(-1, 0, 0),
        int3(0, 1, 0),
        int3(0, -1, 0),
        int3(0, 0, 1),
        int3(0, 0, -1)
    };
    
    uint normalDirection = 0;
    float maxAccCos = 0.0f;
    
    for (uint i = 0; i < 6; i++)
    {
        float currentAccCos = dot(normal, axisDirections[i]);
                    
        if (currentAccCos > maxAccCos)
        {
            maxAccCos = currentAccCos;
            normalDirection = i;
        }
    }
    
    return float3(axisDirections[normalDirection]);
}


[numthreads(8, 8, 8)]
void CS(uint3 GroupId : SV_GroupID, uint GroupIndex : SV_GroupIndex, uint3 GroupThreadID : SV_GroupThreadID)
{
    // We reuse the original TileMap, and at every iteration we reduce the number of tiles by 8 (2 in every dimension).
    // So we need to offset the TileMap by 2 the first iteration, 4 the seconds, 8 the third, etc.
    // We start at 1 because the iteration 0 is the original one for voxelization
    uint tileMapOffset = pow(2, CurrentIteration + 1);
    uint threadLinearIndex = GroupId.x * 512 + GroupIndex;
    
    if (CurrentPhase == 0)
    {        
        uint3 TileID = (GroupId * 8 + GroupThreadID);
        
        if (any(TileID >= TileDimension))
            return;
        
        uint nSubClusterInTile = 0;
            
        uint subClusterIndex = gTileBuffer[TileID];

          
        // At step 0 the sub cluster data is the old super cluster data, so we
        // Read from the super cluster buffer. We only want to know how many clusters are in a tile
        // So we can also reset the tile and next cluster buffer as we go
        while (subClusterIndex != UINT_MAX)
        {
            nSubClusterInTile++;
            uint nextIndex = gNextSubClusterInSuperCluster[subClusterIndex];
            gNextSubClusterInSuperCluster[subClusterIndex] = UINT_MAX;
            subClusterIndex = nextIndex;
        }
        
        gTileBuffer[TileID] = UINT_MAX;


        

        // Minus Epislon to avoid overshooting the maximum number of super clusters
        uint nSuperClustersInTile = ceil(((NumberOfSuperClusters * (float) nSubClusterInTile) / (float) NumberOfSubclusters));
        
        nSuperClustersInTile = (nSuperClustersInTile == 0 && nSubClusterInTile > 0) ? 1 : nSuperClustersInTile;

        
        uint tileClusterGrid = ceil(pow(nSuperClustersInTile, 1 / 3.0f));
        
        float clusterOffset = (2.0f * S) / tileClusterGrid;
        
        // We fill the super cluster buffer with the new super cluster data
        ClusterData data;
        data.VoxelCount = 0;
        data.Normal = float3(1.0f, 1.0f, 1.0f);
        data.FirstDataIndex = UINT_MAX;

        uint originalValue = 0;
        
        uint3 FirstVoxelInTile = TileID * 2 * S;
              
        for (uint nCluster = 0; nCluster < nSuperClustersInTile; nCluster++)
        {
            float3 pos = GetVoxelPosition(nCluster, uint3(tileClusterGrid, tileClusterGrid, tileClusterGrid));

            pos = FirstVoxelInTile + S;

            data.Center = (clamp(round(pos), 0.0f, (float) GridDimension));
            
            InterlockedAdd(gClusterCounterBuffer[0], 1, originalValue);
            
            if (originalValue >= NumberOfSuperClusters)
                return;
            
            gSuperClusterDataBuffer[originalValue] = data;
            gNextSuperClusterInTile[originalValue] = UINT_MAX;
        }

    }
    else if (CurrentPhase == 1)
    {
        
        // This thread is called with NumberOfClusters / 512 groups of 512 threads.
        // So the total number of threads is NumberOfClusters
        uint subClustersPerThread = ceil((float) NumberOfSubclusters / (512.0f * ceil(NumberOfSuperClusters / 512.0f)));
        
        uint startSubIdx = threadLinearIndex * subClustersPerThread;
        uint endSubIdx = min(startSubIdx + subClustersPerThread, NumberOfSubclusters);
        
        for (uint subIdx = 0; subIdx < endSubIdx; subIdx++)
        {
            gSubClusterAssignmentBuffer[subIdx] = UINT_MAX;
        }
        
        if (threadLinearIndex >= NumberOfSuperClusters)
            return;
        
        ClusterData superClusterData = gSuperClusterDataBuffer[threadLinearIndex];
        
        uint subClusterIndex = superClusterData.FirstDataIndex;
        
        float3 avgPos = float3(0.0f, 0.0f, 0.0f);
        float3 avgNormal = float3(0.0f, 0.0f, 0.0f);
        uint subClusterCount = 0;
        float3 lastAvg = float3(0.0f, 0.0f, 0.0f);
        
        while (subClusterIndex != UINT_MAX)
        {
            ClusterData subClusterData = gSubClusterDataBuffer[subClusterIndex];
            avgPos += subClusterData.Center;
            avgNormal += subClusterData.Normal;
            subClusterCount++;
            
            if (!(all(avgNormal > -EPSILON) && all(avgNormal < EPSILON)))
            {
                lastAvg = avgNormal;
            }
            
            // Get the next sub cluster
            uint nextSubIndex = gNextSubClusterInSuperCluster[subClusterIndex];
            gNextSubClusterInSuperCluster[subClusterIndex] = UINT_MAX;
            subClusterIndex = nextSubIndex;
        }
        
        if (FirstClusterSet > 0 && subClusterCount > 0)
        {
            superClusterData.Center = avgPos / subClusterCount;
            superClusterData.Normal = normalize(lastAvg);

        }
        superClusterData.Center = clamp(round(superClusterData.Center), 0.0f, (float) GridDimension);
        superClusterData.VoxelCount = subClusterCount;
        superClusterData.FirstDataIndex = UINT_MAX;
        
        gSuperClusterDataBuffer[threadLinearIndex] = superClusterData;
        
        uint3 tileCoord = (uint3) floor(superClusterData.Center / (2 * S));
        
        uint prev = UINT_MAX;
        uint currentValue;
        InterlockedCompareExchange(gTileBuffer[tileCoord], prev, threadLinearIndex, currentValue);
        
        [allow_uav_condition]
        while (currentValue != prev)
        {
            prev = currentValue;
            gNextSuperClusterInTile[threadLinearIndex] = currentValue;
            InterlockedCompareExchange(gTileBuffer[tileCoord], prev, threadLinearIndex, currentValue);
        }
 
    }
    else if (CurrentPhase == 2)
    {

        if (threadLinearIndex >= NumberOfSubclusters)
            return;
      
        float minDistance = UINT_MAX;
        uint closestClusterIndex = UINT_MAX;
        
        ClusterData subClusterData = gSubClusterDataBuffer[threadLinearIndex];
        int3 subClusterTile = int3(floor(subClusterData.Center / (2 * S)));
        
        int offset = (int) tileMapOffset;
        float maxDistance = (float) tileMapOffset * S;
        float fraction = (m / S);
        
        bool noCloseAngle = true;
        float minSpatialDistance = UINT_MAX;
        
        for (int i = -offset; i <= offset; i++)
        {
            for (int j = -offset; j <= offset; j++)
            {
                for (int k = -offset; k <= offset; k++)
                {
                    int3 tileCoord = subClusterTile + int3(i, j, k);
                    
                    if (any(tileCoord < 0) || any(tileCoord >= int3(TileDimension)))
                        continue;
                    
                    uint clusterIndex = gTileBuffer[tileCoord];
                    

        
                    while (clusterIndex != UINT_MAX)
                    {
                        ClusterData superClusterData = gSuperClusterDataBuffer[clusterIndex];
                        
                        
                        float3 d = float3(abs(subClusterData.Center.x - superClusterData.Center.x),
                            abs(subClusterData.Center.y - superClusterData.Center.y),
                            abs(subClusterData.Center.z - superClusterData.Center.z));
                        
                        if (any(d) > maxDistance)
                        {
                            clusterIndex = gNextSuperClusterInTile[clusterIndex];
                            continue;
                        }


                        float dotProduct = superClusterData.VoxelCount > 0 ? dot(subClusterData.Normal, superClusterData.Normal) : cos30 + EPSILON;
                        
                        
                        float distance = fraction * (d.x + d.y + d.z);
    
                        if (dotProduct > cos30)
                        {
                            dotProduct = 6.0f * S * (1.0f - dotProduct);
                            distance += dotProduct;
                            
                            if (distance < minDistance)
                            {
                                minDistance = distance;
                                closestClusterIndex = clusterIndex;
                            }
                            
                            noCloseAngle = false;
                        }
                        else if (noCloseAngle && distance < minSpatialDistance)
                        {
                            minSpatialDistance = distance;
                            closestClusterIndex = clusterIndex;
                        }
                        else if (closestClusterIndex == UINT_MAX)
                        {
                            closestClusterIndex = clusterIndex;
                        }
                        

                        clusterIndex = gNextSuperClusterInTile[clusterIndex];
                    }
                    
                }
            }
        }
            
        gSubClusterAssignmentBuffer[threadLinearIndex] = closestClusterIndex;
        
        if (closestClusterIndex == UINT_MAX)
        {
            return;
        }
        
        
        uint prev = UINT_MAX;
        uint currentValue;
        InterlockedCompareExchange(gSuperClusterDataBuffer[closestClusterIndex].FirstDataIndex, prev, threadLinearIndex, currentValue);

                        
        [allow_uav_condition]
        while (currentValue != prev)
        {
            prev = currentValue;
            gNextSubClusterInSuperCluster[threadLinearIndex] = currentValue;
            InterlockedCompareExchange(gSuperClusterDataBuffer[closestClusterIndex].FirstDataIndex, prev, threadLinearIndex, currentValue);
        }
    }
    else if (CurrentPhase == 3)
    {

        if (threadLinearIndex >= VoxelCount)
            return;
          
        if (threadLinearIndex < TileDimension.x * TileDimension.y * TileDimension.z)
        {
            uint3 tileId = GetVoxelPosition(threadLinearIndex, TileDimension);
            gTileBuffer[tileId] = UINT_MAX;
        }
        
        if (threadLinearIndex < NumberOfSuperClusters)
        {
            gNextSuperClusterInTile[threadLinearIndex] = UINT_MAX;
        }
    }
    else if (CurrentPhase == 4)
    {
       
        if (threadLinearIndex == 0)
        {
            gClusterCounterBuffer[0] = 0;
        }
        
        //uint totalThreads = ceil((float) NumberOfSuperClusters / 512.0f) * 512;
        //uint voxelPerThread = ceil((float) VoxelCount / totalThreads);
        
        //uint startVoxel = threadLinearIndex * voxelPerThread;
        //uint endVoxel = min(startVoxel + voxelPerThread, VoxelCount);
        
        //for (uint i = startVoxel; i < endVoxel; i++)
        //{
        //    uint prevAssignment = gVoxelAssignmentBuffer[i];
        
        //    if (prevAssignment != UINT_MAX)
        //    {
        //        gVoxelAssignmentBuffer[i] = gSubClusterAssignmentBuffer[prevAssignment];
        //    }
        //}
        
        if (threadLinearIndex >= NumberOfSuperClusters)
            return;
        
        ClusterData cData = gSuperClusterDataBuffer[threadLinearIndex];
            
        uint subDataIndex = cData.FirstDataIndex;
        uint nVoxels = 0;
        float3 posAverage = float3(0.0f, 0.0f, 0.0f);
        float3 normalAverage = float3(0.0f, 0.0f, 0.0f);
        float3 lastNormal = float3(0.0f, 0.0f, 0.0f);
        
        while (subDataIndex != UINT_MAX)
        {
            ClusterData subClusterData = gSubClusterDataBuffer[subDataIndex];
            
            posAverage += subClusterData.Center;
            normalAverage += subClusterData.Normal;
            
            
            if (!(all(normalAverage > -EPSILON) && all(normalAverage < EPSILON)))
            {
                lastNormal = normalAverage;
            }

            subDataIndex = gNextSubClusterInSuperCluster[subDataIndex];
            
            nVoxels += 1;
        }
            
        if (nVoxels > 0)
        {
            cData.Center = posAverage / nVoxels;
            cData.Normal = normalize(lastNormal);
        }
        cData.Center = clamp(round(cData.Center), 0.0f, (float) GridDimension);
        cData.VoxelCount = nVoxels;

            
        gSuperClusterDataBuffer[threadLinearIndex] = cData;
                
    }
    else if (CurrentPhase == 5)
    {
        
        if (threadLinearIndex >= NumberOfSuperClusters)
            return;
        
        ClusterData cData = gSuperClusterDataBuffer[threadLinearIndex];
        
        if (cData.VoxelCount < 1)
        {
            gSuperClusterDataBuffer[threadLinearIndex].FirstDataIndex = UINT_MAX;
            return;
        }

        
        uint originalValue = 0;
        InterlockedAdd(gClusterCounterBuffer[0], 1, originalValue);
        
        uint subClusterIndex = cData.FirstDataIndex;
        
        while (subClusterIndex != UINT_MAX)
        {          
            gSubClusterAssignmentBuffer[subClusterIndex] = originalValue;
            uint nextIndex = gNextSubClusterInSuperCluster[subClusterIndex];
            gNextSubClusterInSuperCluster[subClusterIndex] = UINT_MAX;
            subClusterIndex = nextIndex;
        }
        
        cData.FirstDataIndex = UINT_MAX;
        
        gSubClusterDataBuffer[originalValue] = cData;
    }
    else if (CurrentPhase == 6)
    {      
        if (threadLinearIndex >= VoxelCount)
            return;
        
        uint subIndex = gVoxelAssignmentBuffer[threadLinearIndex];
        
        if (subIndex != UINT_MAX)
        {
            gVoxelAssignmentBuffer[threadLinearIndex] = gSubClusterAssignmentBuffer[subIndex];
        }
        
        if (threadLinearIndex >= gClusterCounterBuffer[0])
            return;
        
        ClusterData cData = gSubClusterDataBuffer[threadLinearIndex];
        
        uint3 tileCoord = (uint3) floor(cData.Center / (2 * S));
        tileCoord = tileCoord * tileMapOffset;
            
        uint prev = UINT_MAX;
        uint currentValue;
        InterlockedCompareExchange(gTileBuffer[tileCoord], prev, threadLinearIndex, currentValue);
        
        [allow_uav_condition]
        while (currentValue != prev)
        {
            prev = currentValue;
            gNextSubClusterInSuperCluster[threadLinearIndex] = currentValue;
            InterlockedCompareExchange(gTileBuffer[tileCoord], prev, threadLinearIndex, currentValue);
        }
    }
    else if (CurrentPhase == 7)
    {
        if (threadLinearIndex >= NumberOfSubclusters)
            return;
        
        ClusterData cData = gSubClusterDataBuffer[threadLinearIndex];
        cData.FirstDataIndex = UINT_MAX;
        cData.VoxelCount = 0;
        
        gSuperClusterDataBuffer[threadLinearIndex] = cData;
    }
    else if (CurrentPhase == 8)
    {
        if (threadLinearIndex >= VoxelCount)
            return;
        
        uint clusterIndex = gVoxelAssignmentBuffer[threadLinearIndex];
        
        if (clusterIndex == UINT_MAX)
            return;
        
        // Linked list of voxels in the cluster
        uint prev = UINT_MAX;
        uint currentValue;
        InterlockedCompareExchange(gSuperClusterDataBuffer[clusterIndex].FirstDataIndex, prev, threadLinearIndex, currentValue);
        
        [allow_uav_condition]
        while (currentValue != prev)
        {
            prev = currentValue;
            gNextVoxelBuffer[threadLinearIndex] = currentValue;
            InterlockedCompareExchange(gSuperClusterDataBuffer[clusterIndex].FirstDataIndex, prev, threadLinearIndex, currentValue);
        }
        
        InterlockedAdd(gSuperClusterDataBuffer[clusterIndex].VoxelCount, 1);
    }

}