#include "VoxelUtils.hlsli"

cbuffer SLICCommon : register(b0)
{
    uint CurrentPhase;
    uint VoxelCount;
    uint K; // Number of clusters
    float m; // Compactness factor
    
    uint3 GridDimension;
    uint S; // Cluster area
    
    uint3 TileDimension;
    uint FirstClusterSet;
    
    uint3 TilesToUpdate;
    uint UnassignedOnlyPass;
}


StructuredBuffer<FragmentData> gFragmentBuffer : register(t0, space0);
StructuredBuffer<uint> gNextIndexBuffer : register(t1, space0);

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space1);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space1);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space1);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space1);


// Buffer containing the cluster data used by this shader
// Dim: K
RWStructuredBuffer<ClusterData> gClusterDataBuffer : register(u0, space0);

// Buffer representing the final voxel assignment for each cluster, as a linked list
// The first element is stored in the field FirstDataIndex in the ClusterData struct, and
// each element is the index of the next element in the list 
// e.g. gClusterDataBuffer[1002].FirstDataIndex = 2, gNextVoxelLinkedList[2] = 2013 gNextVoxelLinkedList[2013] = UINT_MAX
// means that cluster 1002 has voxels 2 and 2013 assigned to it.
// Dim: VoxelCount
RWStructuredBuffer<uint> gNextVoxelLinkedList: register(u1, space0);

// Buffer representing a mapping for voxels assigned to a cluster. This represents the same voxel assignment as the previous buffer,
// But while the previous one is a linked list, this one is a direct mapping. The previous one is useful for retrieving all the voxels
// assigned for a given cluster idx, this one is useful for retrieving the cluster idx for a given voxel idx.
// e.g. gVoxelAssigmnetMap[2] = 1002 means that voxel 2 is assigned to cluster 1002
// Dim: VoxelCount
RWStructuredBuffer<uint> gVoxelAssignmentMap : register(u2, space0);

// 3D texture representing the subdivision of the space. Each element (Tile) represents a size of the space equal to (2S x 2S x 2S)
// Dim: (GridDimension.x / 2S) x (GridDimension.y / 2S) x (GridDimension.z / 2S)
RWTexture3D<uint> gTileBuffer : register(u3, space0);

// Buffer representing the linked list of clusters in each tile. Each element is the index of the next cluster in the same tile.
// The first cluster is stored in the TileBuffer.
// e.g. gTileBuffer[0,0,0] = 10, gNextClusterInTileLinkedList[10] = 21, gNextClusterInTileLinkedList[21] = UINT_MAX
// means that at the tile (0,0,0) we have clusters 10 and 21.
// Dim: K
RWStructuredBuffer<uint> gNextClusterInTileLinkedList : register(u4, space0);

// Generic counter buffer
// Dim: 1
RWStructuredBuffer<uint> gCounter : register(u5, space0);

// Buffer containing the normal direction of each voxel. All the normal directions are set to one of the axis positive or negative directions
// Depending on the nearby empty voxels and the normal direction of the fragments in the voxel
// Dim: VoxelCount
RWStructuredBuffer<float3> gVoxelNormalDirectionBuffer : register(u6, space0);

// Buffer used to store cluster data, used as a temporary buffer for multi threaded operations
// Will also be used on the next step for cluster merging
RWStructuredBuffer<ClusterData> gSubclusterDataBuffer : register(u7, space0);



uint2 FindHash(uint index, uint rank, uint hashedPosition)
{
    uint2 result = uint2(0, 0);
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

uint2 FindHashedCompactedPositionIndex(uint3 coord, uint3 gridDimension)
{
    uint2 result = uint2(0, 0); // y field is control value, 0 means element not found, 1 means element found
    uint indirectionIndex = gridDimension.z * coord.z + coord.y;
    uint index = gIndirectionIndexBuffer[indirectionIndex];
    uint rank = gIndirectionRankBuffer[indirectionIndex];
    uint hashedPosition = GetLinearCoord(coord, gridDimension);
    
    if (rank > 0 && any(coord < GridDimension))
    {
        return FindHash(index, rank, hashedPosition);
    }
    // This unnecessary else statement is here because the HLSL compiler doesn't like early returns. This is a workaround to avoid
    // Getting compiler warnings
    else
    {
        return result;
    }
}

uint GetEmptyVoxelX(uint3 voxelCoord)
{
    int3 negNeighbour = int3(voxelCoord.x - 1, voxelCoord.y, voxelCoord.z);
    int3 posNeighbour = int3(voxelCoord.x + 1, voxelCoord.y, voxelCoord.z);
    
    uint emptyVoxels = 0;

    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            int3 adjacentVoxel = negNeighbour + int3(0, i, j);
            
            if (any(adjacentVoxel < 0) || any(adjacentVoxel >= int3(GridDimension)))
            {
                emptyVoxels++;
            }
            else
            {
                emptyVoxels += 1 - FindHashedCompactedPositionIndex(uint3(adjacentVoxel), GridDimension).y;
            }
            
            adjacentVoxel = posNeighbour + int3(0, i, j);
            
            if (any(adjacentVoxel < 0) || any(adjacentVoxel >= int3(GridDimension)))
            {
                emptyVoxels++;
            }
            else
            {
                emptyVoxels += 1 - FindHashedCompactedPositionIndex(uint3(adjacentVoxel), GridDimension).y;
            }
        }
    }
    
    return emptyVoxels;
}

uint GetEmptyVoxelY(uint3 voxelCoord)
{
    int3 negNeighbour = int3(voxelCoord.x, voxelCoord.y - 1, voxelCoord.z);
    int3 posNeighbour = int3(voxelCoord.x, voxelCoord.y + 1, voxelCoord.z);
    
    uint emptyVoxels = 0;

    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            int3 adjacentVoxel = negNeighbour + int3(i, 0, j);
            
            if (any(adjacentVoxel < 0) || any(adjacentVoxel >= int3(GridDimension)))
            {
                emptyVoxels++;
            }
            else
            {
                emptyVoxels += 1 - FindHashedCompactedPositionIndex(uint3(adjacentVoxel), GridDimension).y;
            }
            
            adjacentVoxel = posNeighbour + int3(i, 0, j);
            
            if (any(adjacentVoxel < 0) || any(adjacentVoxel >= int3(GridDimension)))
            {
                emptyVoxels++;
            }
            else
            {
                emptyVoxels += 1 - FindHashedCompactedPositionIndex(uint3(adjacentVoxel), GridDimension).y;
            }
        }
    }
    
    return emptyVoxels;
}

uint GetEmptyVoxelZ(uint3 voxelCoord)
{
    int3 negNeighbour = int3(voxelCoord.x, voxelCoord.y, voxelCoord.z - 1);
    int3 posNeighbour = int3(voxelCoord.x, voxelCoord.y, voxelCoord.z + 1);
    
    uint emptyVoxels = 0;

    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            int3 adjacentVoxel = negNeighbour + int3(i, j, 0);
            
            if (any(adjacentVoxel < 0) || any(adjacentVoxel >= int3(GridDimension)))
            {
                emptyVoxels++;
            }
            else
            {
                emptyVoxels += 1 - FindHashedCompactedPositionIndex(uint3(adjacentVoxel), GridDimension).y;
            }
            
            adjacentVoxel = posNeighbour + int3(i, j, 0);
            
            if (any(adjacentVoxel < 0) || any(adjacentVoxel >= int3(GridDimension)))
            {
                emptyVoxels++;
            }
            else
            {
                emptyVoxels += 1 - FindHashedCompactedPositionIndex(uint3(adjacentVoxel), GridDimension).y;
            }
        }
    }
    
    return emptyVoxels;
}

float GetAccumulatedDot(uint index, float3 direction)
{
    float sum = 0.0f;
    uint fragmentIndex = gVoxelIndicesCompactBuffer[index];
            
    while (fragmentIndex != UINT_MAX)
    {
        sum += acos(dot(normalize(gFragmentBuffer[fragmentIndex].normal), direction));
        fragmentIndex = gNextIndexBuffer[fragmentIndex];
    }
    
    return sum;
}

float GetClusterNormalDirection(uint clusterIndex, float3 axisDirection)
{
    ClusterData cData = gClusterDataBuffer[clusterIndex];
    
    float sum = 0.0f;
    
    uint voxelDataIndex = cData.FirstDataIndex;
    while (voxelDataIndex != UINT_MAX)
    {
        sum += acos(dot(gVoxelNormalDirectionBuffer[voxelDataIndex], axisDirection));
        voxelDataIndex = gNextVoxelLinkedList[voxelDataIndex];
    }
    
    return sum;

}

void SetUpVoxelNormal(uint hashIndex)
{  
    if (hashIndex >= VoxelCount)
        return;
        
    int3 axisDirections[6] =
    {
        int3(1, 0, 0),
        int3(-1, 0, 0),
        int3(0, 1, 0),
        int3(0, -1, 0),
        int3(0, 0, 1),
        int3(0, 0, -1)
    };
    
    uint3 voxelPos = GetVoxelPosition(gVoxelHashedCompactBuffer[hashIndex], GridDimension);
    
    uint maxEmptyVoxels = 0;
    uint emptyVoxelCount[3];
    
    emptyVoxelCount[0] = GetEmptyVoxelX(voxelPos);
    emptyVoxelCount[1] = GetEmptyVoxelY(voxelPos);
    emptyVoxelCount[2] = GetEmptyVoxelZ(voxelPos);
    
    for (uint j = 0; j < 3; j++)
    {
        maxEmptyVoxels = max(maxEmptyVoxels, emptyVoxelCount[j]);
    }
    
    uint normalDirection = 4;


    float minSum = UINT_MAX;
    uint nOfSame = 0;
    
    for (uint i = 0; i < 6; i++)
    {
        if (emptyVoxelCount[i / 2] == maxEmptyVoxels)
        {
            float currentAccCos = GetAccumulatedDot(hashIndex, float3(axisDirections[i]));
                    
            if (currentAccCos < minSum)
            {
                minSum = currentAccCos;
                normalDirection = i;
            }
        }
    }

    gVoxelNormalDirectionBuffer[hashIndex] = float3(axisDirections[normalDirection]);
    
}

void ComputeDistance(ClusterData cData, float3 voxelCoord, float3 voxelNormal, out float3 distance, out float dotProduct)
{
    float3 d = float3(abs(cData.Center.x - voxelCoord.x),
                          abs(cData.Center.y - voxelCoord.y),
                          abs(cData.Center.z - voxelCoord.z));
    
    float dotP = cData.VoxelCount > 0 ? dot(voxelNormal, cData.Normal) : cos30 + EPSILON;
    
    dotProduct = dotP;
    distance = d;   
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
        avgPos += float3(GetVoxelPosition(gVoxelHashedCompactBuffer[idx], GridDimension));
        avgNormal += gVoxelNormalDirectionBuffer[idx];
        
        if (!(all(avgNormal < EPSILON) && all(avgNormal > -EPSILON)))
        {
            lastAvgNormal = avgNormal;
        }
                
        count += 1;
                
        uint nextIndex = gNextVoxelLinkedList[idx];
        gNextVoxelLinkedList[idx] = emptyList ? UINT_MAX : nextIndex;
        idx = nextIndex;
    }
    
    averagePosition = avgPos / max(count, 1);
    averageNormal = normalize(lastAvgNormal);
    numberOfVoxels = count;
}

[numthreads(8, 8, 8)]
void CS(uint3 GridID : SV_GroupID, uint GroupThreadIndex : SV_GroupIndex, uint3 GroupThreadID : SV_GroupThreadID)
{   
    uint3 GridSize = ceil(TileDimension / (8.0f)); // 8 * 8 * 8;
    
    uint threadLinearIndex = GridID.z * (GridSize.x * GridSize.y) + GridID.y * GridSize.x + GridID.x;
    threadLinearIndex = threadLinearIndex * 512 + GroupThreadIndex;

    uint3 TileID = GridID * 8 + GroupThreadID;
    
    // Called with (VoxelCount / 512, 1, 1) thread groups
    if (CurrentPhase == 0)
    {
        if (threadLinearIndex == 0)
        {
            gCounter[0] = 0;
        }
            
        if (threadLinearIndex >= VoxelCount)
            return;
        
        // Precompute normal direction
        SetUpVoxelNormal(threadLinearIndex);
        // Set the cluster assignment to UINT_MAX, which means no assignment
        gNextVoxelLinkedList[threadLinearIndex] = UINT_MAX;
        
        if (threadLinearIndex < K)
        {
            gSubclusterDataBuffer[threadLinearIndex].FirstDataIndex = UINT_MAX;
        }
    }
    // Called with (TileDimension.x / 8, TileDimension.y / 8, TileDimension.z / 8) thread groups
    else if (CurrentPhase == 1)
    {
        if (any(TileID >= TileDimension))
            return;
        
        uint voxelInTileCount = 0;

        uint3 TileStart = TileID * 2 * S;
        uint3 TileEnd = min(TileStart + 2 * S, GridDimension);
        
        for (uint i = TileStart.x; i < TileEnd.x; i++)
        {
            for (uint j = TileStart.y; j < TileEnd.y; j++)
            {
                for (uint k = TileStart.z; k < TileEnd.z; k++)
                {
                    uint2 hashedPositionIndex = FindHashedCompactedPositionIndex(uint3(i, j, k), GridDimension);
                    voxelInTileCount += hashedPositionIndex.y;
                }
            }
        }
        
        gTileBuffer[TileID] = UINT_MAX;
        

        // Minus Epislon to avoid overshooting K
        uint numberOfClusterInTile = round(((K * (float) voxelInTileCount) / VoxelCount) - EPSILON);
        
        numberOfClusterInTile = (numberOfClusterInTile == 0 && voxelInTileCount > 0) ? 1 : numberOfClusterInTile;

        uint voxelsPerCluster =  voxelInTileCount / numberOfClusterInTile;
        uint nVoxel = 0;
        
        ClusterData data;
        data.VoxelCount = 0;
        data.Normal = float3(1.0f, 1.0f, 1.0f);
        data.FirstDataIndex = UINT_MAX;
        
        // Create a number of clusters per tile proportional to the number of voxels in the same tile.
        // Spread the clusters along the voxels in the same tile and give them a default normal.
        for (uint x = TileStart.x; x < TileEnd.x; x++)
        {
            for (uint y = TileStart.y; y < TileEnd.y; y++)
            {
                for (uint z = TileStart.z; z < TileEnd.z; z++)
                {
                    uint2 hashedPositionIndex = FindHashedCompactedPositionIndex(uint3(x, y, z), GridDimension);
                    nVoxel += hashedPositionIndex.y;
                    
                    if (nVoxel >= voxelsPerCluster)
                    {
                        nVoxel = 0;
                        
                        // We check that the ClusterDataBuffer is not full
                        uint originalValue = 0;
                        InterlockedAdd(gCounter[0], 1, originalValue);
                        
                        if (originalValue < K)
                        {
                            data.Center = float3(GetVoxelPosition(gVoxelHashedCompactBuffer[hashedPositionIndex.x], GridDimension));
                            gClusterDataBuffer[originalValue] = data;
                            gNextClusterInTileLinkedList[originalValue] = UINT_MAX;
                        }

                    }
                }
            }
        }
    }
    // Called with (TileDimension.x / 8, TileDimension.y / 8, TileDimension.z / 8) thread groups
    else if (CurrentPhase == 2)
    {
        uint NumberOfThreads = GridSize.x * GridSize.y * GridSize.z * 512;
        uint voxelsPerThread = ceil((float) VoxelCount / NumberOfThreads);
        uint clusterPerThread = ceil((float) K / (NumberOfThreads));
       
        uint initialindex = threadLinearIndex * voxelsPerThread;
        uint finalindex = min(initialindex + voxelsPerThread, VoxelCount);
        
        for (uint i = initialindex; i < finalindex; i++)
        {
            gVoxelAssignmentMap[i] = UINT_MAX;
        }
        
        initialindex = threadLinearIndex * clusterPerThread;
        finalindex = min(initialindex + clusterPerThread, K);
        
        for (uint j = initialindex; j < finalindex; j++)
        {
            ClusterData cData = gClusterDataBuffer[j];
            
            uint voxelDataIndex = cData.FirstDataIndex;
            uint nVoxels = 0;
            float3 posAverage = float3(0.0f, 0.0f, 0.0f);
            float3 normalAverage = float3(0.0f, 0.0f, 0.0f);
            
            IterateLinkedList(voxelDataIndex, true, posAverage, normalAverage, nVoxels);
            
                     
            if (FirstClusterSet > 0 && nVoxels > 0)
            {
                cData.Center = posAverage;
            }
            
            cData.Normal = normalAverage;
            cData.VoxelCount = nVoxels;
            cData.FirstDataIndex = UINT_MAX;
            cData.Center = clamp(round(cData.Center), 0.0f, (float) GridDimension);
            
            gClusterDataBuffer[j] = cData;
            
            uint3 tileCoord = (uint3) floor(cData.Center / (2 * S));
            
            uint prev = UINT_MAX;
            uint currentValue;
            InterlockedCompareExchange(gTileBuffer[tileCoord], prev, j, currentValue);
        
            [allow_uav_condition]
            while (currentValue != prev)
            {
                prev = currentValue;
                gNextClusterInTileLinkedList[j] = currentValue;
                InterlockedCompareExchange(gTileBuffer[tileCoord], prev, j, currentValue);
            }
        }

    }
    // Called with (VoxelCount / 512, 1, 1) thread groups
    else if (CurrentPhase == 3)
    {
        // Only tiles which are not adjacent are updated (corner tiles are considered adjacent)
        // So we check if the tile is even or odd for each axis
         
        if (threadLinearIndex >= VoxelCount)
            return;
        
        if (UnassignedOnlyPass == 1)  
        {
            if (gVoxelAssignmentMap[threadLinearIndex] != UINT_MAX)
                return;
        }
        
        uint3 voxelCoord = GetVoxelPosition(gVoxelHashedCompactBuffer[threadLinearIndex], GridDimension);
        
        uint3 voxelTile = (uint3) floor(voxelCoord / (2 * S));
        
        // Returns -1 or 1 for each axis depending on which half of the tile the voxel is in
        int3 voxelHalfInTile = round((voxelCoord - (voxelTile * 2.0f * S)) / (2.0f * S)) * 2 - 1;
        
        // Since the cluster are as big as the tile, worst case scenario is 7 adjacent tile + the one the voxel is in
        // Worst case scenario is when a voxel is in one of the corners of the tile. In this case the adjacent tiles
        // Are all the tiles adjacent to that angle (even if they are only adjacent by a corner)
        int3 adjacentTiles[8];
        uint adjacentTileCount = 0;
        
        
        for (int i = 0; i <= 1; i++)
        {
            for (int j = 0; j <= 1; j++)
            {
                for (int k = 0; k <= 1; k++)
                {
                    
                    int3 adjTile = voxelTile + int3(i, j, k) * voxelHalfInTile;
                    
                    if (any(adjTile < 0) || any(adjTile >= int3(TileDimension)))
                        continue;
                    
                    adjacentTiles[adjacentTileCount] = adjTile;
                    adjacentTileCount++;

                }
            }
        }
        
        float fraction = (m / S);
        float minDistance = UINT_MAX;
        uint closestClusterIndex = UINT_MAX;
        
        float3 voxelNormal = gVoxelNormalDirectionBuffer[threadLinearIndex];
        
        for (uint t = 0; t < adjacentTileCount; t++)
        {
            uint clusterIndex = gTileBuffer[adjacentTiles[t]];
            
            while (clusterIndex != UINT_MAX)
            {
                ClusterData cData = gClusterDataBuffer[clusterIndex];
                
                float3 dis;
                float dotProd;
                ComputeDistance(cData, voxelCoord, voxelNormal, dis, dotProd);

                if (any(dis > 1.0f * S))
                {
                    clusterIndex = gNextClusterInTileLinkedList[clusterIndex];
                    continue;
                }
                
                float distance = fraction * (dis.x + dis.y + dis.z);
                
                if (dotProd > cos30 || ((FirstClusterSet == 0 || UnassignedOnlyPass == 1) && closestClusterIndex == UINT_MAX))
                {           
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                        closestClusterIndex = clusterIndex;
                    }
                    
                }

                clusterIndex = gNextClusterInTileLinkedList[clusterIndex];
            }
        }
        
        gVoxelAssignmentMap[threadLinearIndex] = closestClusterIndex;
        
        uint prev = UINT_MAX;
        uint currentValue;
        InterlockedCompareExchange(gClusterDataBuffer[closestClusterIndex].FirstDataIndex, prev, threadLinearIndex, currentValue);

                        
        [allow_uav_condition]
        while (currentValue != prev)
        {
            prev = currentValue;
            gNextVoxelLinkedList[threadLinearIndex] = currentValue;
            InterlockedCompareExchange(gClusterDataBuffer[closestClusterIndex].FirstDataIndex, prev, threadLinearIndex, currentValue);
        }
             
    }
    // Called with (VoxelCount / 512, 1, 1) thread groups
    else if (CurrentPhase == 4)
    {

        if (threadLinearIndex >= VoxelCount)
            return;
           
        if (threadLinearIndex < TileDimension.x * TileDimension.y * TileDimension.z)
        {
            uint3 tileId = GetVoxelPosition(threadLinearIndex, TileDimension);
            gTileBuffer[tileId] = UINT_MAX;
        }

        if (threadLinearIndex >= K)
            return;
        
        gNextClusterInTileLinkedList[threadLinearIndex] = UINT_MAX;
        
        if (threadLinearIndex == 0)
        {
            gCounter[0] = 0;
        }
    }
    // Called with (K / 512, 1, 1) thread groups
    else if (CurrentPhase == 5)
    {
        if (threadLinearIndex >= K)
            return;
        
        ClusterData cData = gClusterDataBuffer[threadLinearIndex];
            
        uint voxelDataIndex = cData.FirstDataIndex;
        uint nVoxels = 0;
        float3 posAverage = float3(0.0f, 0.0f, 0.0f);
        float3 normalAverage = float3(0.0f, 0.0f, 0.0f);
            
        IterateLinkedList(voxelDataIndex, false, posAverage, normalAverage, nVoxels);
        
        if (nVoxels < 8)
        {
            uint idx = cData.FirstDataIndex;
            while (idx != UINT_MAX)
            {
                gVoxelAssignmentMap[idx] = UINT_MAX;
                idx = gNextVoxelLinkedList[idx];
            }
            
            cData.FirstDataIndex = UINT_MAX;
            cData.VoxelCount = 0;
            gClusterDataBuffer[threadLinearIndex] = cData;
            return;
        }
        
        uint originalValue = 0;
        InterlockedAdd(gCounter[0], 1, originalValue);
        
        if (originalValue >= K)
            return;
        
        uint voxIdx = cData.FirstDataIndex;
        while (voxIdx != UINT_MAX)
        {
            gVoxelAssignmentMap[voxIdx] = originalValue;
            voxIdx = gNextVoxelLinkedList[voxIdx];
        }
        
        IterateLinkedList(voxelDataIndex, false, posAverage, normalAverage, nVoxels);
        
        cData.Center = posAverage;
        cData.Normal = normalize(normalAverage);
        cData.VoxelCount = nVoxels;
        gClusterDataBuffer[threadLinearIndex] = cData;
        
        

        
        uint3 tileCoord = (uint3) floor(cData.Center / (2 * S));
        
        // We fill the tile buffer again because we are going to do a final pass to assing orphan voxels
        uint prev = UINT_MAX;
        uint currentValue;
        InterlockedCompareExchange(gTileBuffer[tileCoord], prev, originalValue, currentValue);
        
        [allow_uav_condition]
        while (currentValue != prev)
        {
            prev = currentValue;
            gNextClusterInTileLinkedList[originalValue] = currentValue;
            InterlockedCompareExchange(gTileBuffer[tileCoord], prev, originalValue, currentValue);
        }
        
        // We need a second buffer to store the new cluster data, because if we reuse gClusterDataBuffer, there is no way to guarantee
        // that gClusterDataBuffer[originalValue] is safe to be updated.
        gSubclusterDataBuffer[originalValue] = cData;
    }

}