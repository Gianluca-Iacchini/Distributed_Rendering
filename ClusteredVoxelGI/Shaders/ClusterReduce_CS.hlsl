#include "VoxelUtils.hlsli"

static float CBRT2 = 1.2599210498948731647672106072782f;

cbuffer SLICCommon : register(b0)
{
    uint CurrentPhase;
    uint NumberOfSubclusters;
    uint NumberOfSuperClusters; // Number of clusters
    float m; // Compactness factor
    
    uint3 GridDimension;
    uint VoxelCount;


    uint3 TileDimension;
    uint S; // Cluster area
    
    uint3 PreviousTileDimension;
    uint PreviousS;
    
    uint CurrentSubdivision;
    uint FirstClusterSet;
    float _pad1; // Original number of clusters
    float _pad2;
    

}

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space0);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space0);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space0);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space0);


RWStructuredBuffer<ClusterData> gSuperClusterDataBuffer : register(u0, space0);
RWStructuredBuffer<uint> gNextVoxelLinkedList : register(u1, space0);
RWStructuredBuffer<uint> gVoxelAssignmentBuffer : register(u2, space0);
RWTexture3D<uint> gSuperClusterTileBuffer : register(u3, space0);
RWStructuredBuffer<uint> gNextSuperClusterInTileLinkedList : register(u4, space0);
RWStructuredBuffer<uint> gCounter : register(u5, space0);
RWStructuredBuffer<float3> gVoxelNormalDirectionBuffer : register(u6, space0);
RWStructuredBuffer<ClusterData> gSubClusterDataBuffer : register(u7, space0);

RWStructuredBuffer<uint> gNextSubClusterInSuperCluster : register(u0, space1);
RWStructuredBuffer<uint> gSubClusterAssignmentBuffer : register(u1, space1);
RWTexture3D<uint> gSubClusterTileBuffer : register(u2, space1);
RWStructuredBuffer<uint> gNextSubClusterInTileLInkedList : register(u3, space1);


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

void IterateLinkedList(
uint firstIndex,
bool emptyList,
out float3 averagePosition,
out float3 averageNormal,
out uint numberOfVoxels)
{
    uint idx = firstIndex;
    
    float3 avgPos = float3(0.0f, 0.0f, 0.0f);
    float3 avgNormal = float3(0.0f, 0.0f, 0.0f);
    float3 lastAvgNormal = float3(1.0f, 1.0f, 1.0f);
    
    uint count = 0;
    while (idx != UINT_MAX)
    {
        ClusterData subData = gSubClusterDataBuffer[idx];
        
        if (subData.VoxelCount > 0)
        {

            avgPos += subData.Center;
            avgNormal += subData.Normal;
        
            if (!(all(avgNormal < EPSILON) && all(avgNormal > -EPSILON)))
            {
                lastAvgNormal = avgNormal;
            }
                
            count += 1;
                
            uint nextIndex = gNextSubClusterInSuperCluster[idx];
            gNextSubClusterInSuperCluster[idx] = emptyList ? UINT_MAX : nextIndex;
            idx = nextIndex;
        }
    }
    
    averagePosition = avgPos / max(count, 1);
    averageNormal = normalize(lastAvgNormal);
    numberOfVoxels = count;
}

[numthreads(8, 8, 8)]
void CS(uint3 GroupId : SV_GroupID, uint GroupIndex : SV_GroupIndex, uint3 GroupThreadID : SV_GroupThreadID)
{
    uint3 GridSize = ceil(TileDimension / (8.0f));
    
    uint threadLinearIndex = GroupId.z * (GridSize.x * GridSize.y) + GroupId.y * GridSize.x + GroupId.x;
    threadLinearIndex = threadLinearIndex * 512 + GroupIndex;

    uint3 TileID = GroupId * 8 + GroupThreadID;
    
    // Called with a thread group count of (VoxelCount / 512, 1, 1)
    if (CurrentPhase == 0)
    {      
        if (threadLinearIndex == 0)
        {
            gCounter[0] = 0;
        }
        
        if (threadLinearIndex >= VoxelCount)
            return;
        
        if (threadLinearIndex < TileDimension.x * TileDimension.y * TileDimension.z)
        {
            uint3 tileId = GetVoxelPosition(threadLinearIndex, TileDimension);
            gSuperClusterTileBuffer[tileId] = UINT_MAX;
        }
        
        if (threadLinearIndex < PreviousTileDimension.x * PreviousTileDimension.y * PreviousTileDimension.z)
        {
            uint3 tileId = GetVoxelPosition(threadLinearIndex, PreviousTileDimension);
            gSubClusterTileBuffer[tileId] = UINT_MAX;
        }
        
        // During this pass we have not yet subdivided the clusters. Number of super clusters is therefore equal to the number of clusters
        // And number of subclusters is equal to the number of non empty clusters.
        if (threadLinearIndex >= NumberOfSuperClusters)
            return;
        
        ClusterData data;
        data.Center = float3(0.0f, 0.0f, 0.0f);
        data.Normal = float3(0.0f, 0.0f, 1.0f);
        data.VoxelCount = 0;
        data.FirstDataIndex = UINT_MAX;
        
        gSuperClusterDataBuffer[threadLinearIndex] = data;
        gNextSuperClusterInTileLinkedList[threadLinearIndex] = UINT_MAX;
        
        // Since this is equl to the number of non empty clusters (for this pass only), this is less than or equal NumberOfSuperClusters
        if (threadLinearIndex >= NumberOfSubclusters)
            return;
        
        gNextSubClusterInSuperCluster[threadLinearIndex] = UINT_MAX;
        gSubClusterAssignmentBuffer[threadLinearIndex] = UINT_MAX;
        gNextSubClusterInTileLInkedList[threadLinearIndex] = UINT_MAX;
    }
    // Called with a number of thread groups equal to (NumberOfSubclusters / 512, 1, 1)
    else if (CurrentPhase == 1)
    {
        if (threadLinearIndex >= NumberOfSubclusters)
            return;
        
        ClusterData data = gSubClusterDataBuffer[threadLinearIndex];
        
        if (data.VoxelCount < 1)
        {
            return;
        }
        
        uint3 tileCoord = (uint3) floor(data.Center / (2 * S));
        
        uint prev = UINT_MAX;
        uint currentValue;
        InterlockedCompareExchange(gSubClusterTileBuffer[tileCoord], prev, threadLinearIndex, currentValue);
        
        [allow_uav_condition]
        while (currentValue != prev)
        {
            prev = currentValue;
            gNextSubClusterInTileLInkedList[threadLinearIndex] = currentValue;
            InterlockedCompareExchange(gSubClusterTileBuffer[tileCoord], prev, threadLinearIndex, currentValue);
        }
    }
    // Called with a number of thread groups equal to (TileDimension.x / 8, TileDimension.y / 8, TileDimension.z / 8)
    else if (CurrentPhase == 2)
    {
        if (any(TileID >= TileDimension))
            return;
        
        // FastSlic_CS ends with the gTileBuffer filled with cluster data since it needs it to perform the last pass.
        // We can therefore reuse that data this pass.
        
        uint nSubClusters = 0;
        uint idx = gSubClusterTileBuffer[TileID];
        
        while (idx != UINT_MAX)
        {
            nSubClusters++;
            idx = gNextSubClusterInTileLInkedList[idx];
        }
        
        uint nSuperClustersInTile = round(((NumberOfSuperClusters * (float) nSubClusters) / (float) NumberOfSubclusters) - EPSILON);
        nSuperClustersInTile = (nSuperClustersInTile == 0 && nSubClusters > 0) ? 1 : nSuperClustersInTile;
        

        
        uint nSubPerSups = nSubClusters / nSuperClustersInTile;
        
        ClusterData superData;
        superData.Center = float3(0.0f, 0.0f, 0.0f);
        superData.Normal = float3(0.0f, 0.0f, 1.0f);
        superData.VoxelCount = 0;
        superData.FirstDataIndex = UINT_MAX;
        
        nSubClusters = 0;
       
        idx = gSubClusterTileBuffer[TileID];
        uint originalValue = 0;
        while (idx != UINT_MAX)
        {
            nSubClusters++;
            if (nSubClusters >= nSubPerSups)
            {
                superData.Center = gSubClusterDataBuffer[idx].Center;
                
                InterlockedAdd(gCounter[0], 1, originalValue);
                
                if (originalValue < NumberOfSuperClusters)
                {
                    gSuperClusterDataBuffer[originalValue] = superData;
                    gNextSuperClusterInTileLinkedList[originalValue] = UINT_MAX;
                }
                
                nSubClusters = 0;
            }
            
            idx = gNextSubClusterInTileLInkedList[idx];
        }
    }
    // Called with a number of thread groups equal to (NumberOfSuperClusters / 512, 1, 1)
    else if (CurrentPhase == 3)
    {
        uint numberOfThreads = ceil(NumberOfSuperClusters / 512);
        uint subClustersPerThread = ceil(NumberOfSubclusters / numberOfThreads);
        
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
        
        IterateLinkedList(subClusterIndex, true, avgPos, avgNormal, subClusterCount);
        
        if (FirstClusterSet > 0 && subClusterCount > 0)
        {
            superClusterData.Center = avgPos;
        }
        
        superClusterData.Normal = avgNormal;
        superClusterData.Center = clamp(round(superClusterData.Center), 0.0f, (float) GridDimension);
        superClusterData.VoxelCount = subClusterCount;
        superClusterData.FirstDataIndex = UINT_MAX;
        
        gSuperClusterDataBuffer[threadLinearIndex] = superClusterData;
        
        uint3 tileCoord = (uint3) floor(superClusterData.Center / (2 * S));
        
        uint prev = UINT_MAX;
        uint currentValue;
        InterlockedCompareExchange(gSuperClusterTileBuffer[tileCoord], prev, threadLinearIndex, currentValue);
        
        [allow_uav_condition]
        while (currentValue != prev)
        {
            prev = currentValue;
            gNextSuperClusterInTileLinkedList[threadLinearIndex] = currentValue;
            InterlockedCompareExchange(gSuperClusterTileBuffer[tileCoord], prev, threadLinearIndex, currentValue);
        }
    }
    // Called with a number of thread groups equal to (NumberOfSubclusters / 512, 1, 1)
    else if (CurrentPhase == 4)
    {
        if (threadLinearIndex >= NumberOfSubclusters)
            return;
        
        
    }
    //else if (CurrentPhase == 2)
    //{

    //    if (threadLinearIndex >= NumberOfSubclusters)
    //        return;
      
    //    float minDistance = UINT_MAX;
    //    uint closestClusterIndex = UINT_MAX;
        
    //    ClusterData subClusterData = gSubClusterDataBuffer[threadLinearIndex];
    //    int3 subClusterTile = int3(floor(subClusterData.Center / (2 * S)));
        
    //    // Which adjacent tiles to check. We want to check the tile itself and the surronding tiles in all 3D directions
    //    // (including corners).
    //    // CurrentSubdivisions represent how much we are subdividing the original cluster count, so if we are at CurrentSubdivision = 2
    //    // then we are reducing the number of cluster by OriginalClusterCount / 2.
    //    // This means that each cluster should now check double in volume (since we are using a 3d tile grid), which means that each
    //    // side should grow by pow(cbrt(2),  CurrentSubdivision)
        
    //    float growth = pow(CBRT2, CurrentSubdivision);
    //    int offset = 1 + ceil(growth);
    //    float maxDistance = (float) growth * S;
    //    float fraction = (m / S);
        
    //    bool noCloseAngle = true;
    //    float minSpatialDistance = UINT_MAX;
        
    //    for (int i = -offset; i <= offset; i++)
    //    {
    //        for (int j = -offset; j <= offset; j++)
    //        {
    //            for (int k = -offset; k <= offset; k++)
    //            {
    //                int3 tileCoord = subClusterTile + int3(i, j, k);
                    
    //                if (any(tileCoord < 0) || any(tileCoord >= int3(TileDimension)))
    //                    continue;
                    
    //                uint clusterIndex = gTileBuffer[tileCoord];
                    

        
    //                while (clusterIndex != UINT_MAX)
    //                {
    //                    ClusterData superClusterData = gSuperClusterDataBuffer[clusterIndex];
                        
                        
    //                    float3 d = float3(abs(subClusterData.Center.x - superClusterData.Center.x),
    //                        abs(subClusterData.Center.y - superClusterData.Center.y),
    //                        abs(subClusterData.Center.z - superClusterData.Center.z));
                        
    //                    if (any(d) > maxDistance)
    //                    {
    //                        clusterIndex = gNextSuperClusterInTile[clusterIndex];
    //                        continue;
    //                    }


    //                    float dotProduct = superClusterData.VoxelCount > 0 ? dot(subClusterData.Normal, superClusterData.Normal) : cos30 + EPSILON;
                        
                        
    //                    float distance = fraction * (d.x + d.y + d.z);
    
    //                    if (dotProduct > cos30)
    //                    {
    //                        dotProduct = 6.0f * S * (1.0f - dotProduct);
    //                        distance += dotProduct;
                            
    //                        if (distance < minDistance)
    //                        {
    //                            minDistance = distance;
    //                            closestClusterIndex = clusterIndex;
    //                        }
                            
    //                        noCloseAngle = false;
    //                    }
    //                    else if (noCloseAngle && distance < minSpatialDistance)
    //                    {
    //                        minSpatialDistance = distance;
    //                        closestClusterIndex = clusterIndex;
    //                    }
    //                    else if (closestClusterIndex == UINT_MAX)
    //                    {
    //                        closestClusterIndex = clusterIndex;
    //                    }
                        

    //                    clusterIndex = gNextSuperClusterInTile[clusterIndex];
    //                }
                    
    //            }
    //        }
    //    }
            
    //    gSubClusterAssignmentBuffer[threadLinearIndex] = closestClusterIndex;
        
    //    if (closestClusterIndex == UINT_MAX)
    //    {
    //        return;
    //    }
        
        
    //    uint prev = UINT_MAX;
    //    uint currentValue;
    //    InterlockedCompareExchange(gSuperClusterDataBuffer[closestClusterIndex].FirstDataIndex, prev, threadLinearIndex, currentValue);

                        
    //    [allow_uav_condition]
    //    while (currentValue != prev)
    //    {
    //        prev = currentValue;
    //        gNextSubClusterInSuperCluster[threadLinearIndex] = currentValue;
    //        InterlockedCompareExchange(gSuperClusterDataBuffer[closestClusterIndex].FirstDataIndex, prev, threadLinearIndex, currentValue);
    //    }
    //}
    //else if (CurrentPhase == 3)
    //{

    //    if (threadLinearIndex >= VoxelCount)
    //        return;
          
    //    if (threadLinearIndex < TileDimension.x * TileDimension.y * TileDimension.z)
    //    {
    //        uint3 tileId = GetVoxelPosition(threadLinearIndex, TileDimension);
    //        gTileBuffer[tileId] = UINT_MAX;
    //    }
        
    //    if (threadLinearIndex < NumberOfSuperClusters)
    //    {
    //        gNextSuperClusterInTile[threadLinearIndex] = UINT_MAX;
    //    }
    //}
    //else if (CurrentPhase == 4)
    //{
       
    //    if (threadLinearIndex == 0)
    //    {
    //        gClusterCounterBuffer[0] = 0;
    //    }
        
        
    //    if (threadLinearIndex >= NumberOfSuperClusters)
    //        return;
        
    //    ClusterData cData = gSuperClusterDataBuffer[threadLinearIndex];
            
    //    uint subDataIndex = cData.FirstDataIndex;
    //    uint nVoxels = 0;
    //    float3 posAverage = float3(0.0f, 0.0f, 0.0f);
    //    float3 normalAverage = float3(0.0f, 0.0f, 0.0f);
    //    float3 lastNormal = float3(0.0f, 0.0f, 0.0f);
        
    //    while (subDataIndex != UINT_MAX)
    //    {
    //        ClusterData subClusterData = gSubClusterDataBuffer[subDataIndex];
            
    //        posAverage += subClusterData.Center;
    //        normalAverage += subClusterData.Normal;
            
            
    //        if (!(all(normalAverage > -EPSILON) && all(normalAverage < EPSILON)))
    //        {
    //            lastNormal = normalAverage;
    //        }

    //        subDataIndex = gNextSubClusterInSuperCluster[subDataIndex];
            
    //        nVoxels += 1;
    //    }
            
    //    if (nVoxels > 0)
    //    {
    //        cData.Center = posAverage / nVoxels;
    //        cData.Normal = normalize(lastNormal);
    //    }
    //    cData.Center = clamp(round(cData.Center), 0.0f, (float) GridDimension);
    //    cData.VoxelCount = nVoxels;

            
    //    gSuperClusterDataBuffer[threadLinearIndex] = cData;
                
    //}
    //else if (CurrentPhase == 5)
    //{
        
    //    if (threadLinearIndex >= NumberOfSuperClusters)
    //        return;
        
    //    ClusterData cData = gSuperClusterDataBuffer[threadLinearIndex];
        
    //    if (cData.VoxelCount < 1)
    //    {
    //        cData.FirstDataIndex = UINT_MAX;
    //        cData.VoxelCount = 0;
    //        gSuperClusterDataBuffer[threadLinearIndex] = cData;
    //        return;
    //    }

        
    //    uint originalValue = 0;
    //    InterlockedAdd(gClusterCounterBuffer[0], 1, originalValue);
        
    //    uint subClusterIndex = cData.FirstDataIndex;
        
    //    while (subClusterIndex != UINT_MAX)
    //    {          
    //        gSubClusterAssignmentBuffer[subClusterIndex] = originalValue;
    //        uint nextIndex = gNextSubClusterInSuperCluster[subClusterIndex];
    //        gNextSubClusterInSuperCluster[subClusterIndex] = UINT_MAX;
    //        subClusterIndex = nextIndex;
    //    }
        
    //    cData.FirstDataIndex = UINT_MAX;
        
    //    gSubClusterDataBuffer[originalValue] = cData;
    //}
    //else if (CurrentPhase == 6)
    //{      
    //    if (threadLinearIndex >= VoxelCount)
    //        return;
        
    //    uint subIndex = gVoxelAssignmentBuffer[threadLinearIndex];
        
    //    if (subIndex != UINT_MAX)
    //    {
    //        gVoxelAssignmentBuffer[threadLinearIndex] = gSubClusterAssignmentBuffer[subIndex];
    //    }
        
    //    if (threadLinearIndex >= gClusterCounterBuffer[0])
    //        return;
        
    //    ClusterData cData = gSubClusterDataBuffer[threadLinearIndex];
        
    //    uint3 tileCoord = (uint3) floor(cData.Center / (2 * S));
    //    tileCoord = tileCoord * ceil(pow(CBRT2, CurrentSubdivision));
            
    //    uint prev = UINT_MAX;
    //    uint currentValue;
    //    InterlockedCompareExchange(gTileBuffer[tileCoord], prev, threadLinearIndex, currentValue);
        
    //    [allow_uav_condition]
    //    while (currentValue != prev)
    //    {
    //        prev = currentValue;
    //        gNextSubClusterInSuperCluster[threadLinearIndex] = currentValue;
    //        InterlockedCompareExchange(gTileBuffer[tileCoord], prev, threadLinearIndex, currentValue);
    //    }
    //}
    //else if (CurrentPhase == 7)
    //{
    //    if (threadLinearIndex >= NumberOfSubclusters)
    //        return;
        
    //    ClusterData cData = gSubClusterDataBuffer[threadLinearIndex];
    //    cData.FirstDataIndex = UINT_MAX;
    //    cData.VoxelCount = 0;
        
    //    gSuperClusterDataBuffer[threadLinearIndex] = cData;
    //}
    //else if (CurrentPhase == 8)
    //{
    //    if (threadLinearIndex >= VoxelCount)
    //        return;
        
    //    uint clusterIndex = gVoxelAssignmentBuffer[threadLinearIndex];
        
    //    if (clusterIndex == UINT_MAX)
    //        return;
        
    //    // Linked list of voxels in the cluster
    //    uint prev = UINT_MAX;
    //    uint currentValue;
    //    InterlockedCompareExchange(gSuperClusterDataBuffer[clusterIndex].FirstDataIndex, prev, threadLinearIndex, currentValue);
        
    //    [allow_uav_condition]
    //    while (currentValue != prev)
    //    {
    //        prev = currentValue;
    //        gNextVoxelBuffer[threadLinearIndex] = currentValue;
    //        InterlockedCompareExchange(gSuperClusterDataBuffer[clusterIndex].FirstDataIndex, prev, threadLinearIndex, currentValue);
    //    }
        
    //    InterlockedAdd(gSuperClusterDataBuffer[clusterIndex].VoxelCount, 1);
    //}
}