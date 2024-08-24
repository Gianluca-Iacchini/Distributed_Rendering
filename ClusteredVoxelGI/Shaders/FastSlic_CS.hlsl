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
    float pad1;
}


StructuredBuffer<FragmentData> gFragmentBuffer : register(t0, space0);
StructuredBuffer<uint> gNextIndexBuffer : register(t1, space0);

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space1);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space1);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space1);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space1);


RWStructuredBuffer<ClusterData> gClusterDataBuffer : register(u0, space0);
RWStructuredBuffer<uint> gNextVoxelBuffer: register(u1, space0);
RWStructuredBuffer<uint> gClusterAssignmentBuffer : register(u2, space0);
RWStructuredBuffer<float> gClusterDistanceBuffer : register(u3, space0);
RWTexture3D<uint> gTileBuffer : register(u4, space0);
RWStructuredBuffer<uint> gNextCluster : register(u5, space0);
RWStructuredBuffer<uint> gClusterCounterBuffer : register(u6, space0);
RWStructuredBuffer<float3> gVoxelNormalDirectionBuffer : register(u7, space0);

RWStructuredBuffer<uint> gNextVoxelClusterData : register(u8, space0);


static const float cos30 = 0.81915204428f;



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
    
    if (rank > 0 || any(coord >= GridDimension))
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

uint GetEmptyVoxelCount(uint3 voxelCoord, int3 direction)
{

    // Determine the ranges based on the direction
    int3 rangeMin = int3(0, 0, 0);
    int3 rangeMax = int3(0, 0, 0);
    
    if (direction.x != 0)
    {
        rangeMin = int3(0, -1, -1);
        rangeMax = int3(0, 1, 1);
    }
    else if (direction.y != 0)
    {
        rangeMin = int3(-1, 0, -1);
        rangeMax = int3(1, 0, 1);
    }
    else if (direction.z != 0)
    {
        rangeMin = int3(-1, -1, 0);
        rangeMax = int3(1, 1, 0);
    }
    
    //int3 otherDirection = 2 * abs(direction) - 1;
    
    //int3 offset = direction;
    //offset.x = offset.x == 0 ? 1 : offset.x;
    //offset.y = offset.y == 0 ? 1 : offset.y;
    //offset.z = offset.z == 0 ? 1 : offset.z;
    
    uint emptyVoxels = 0;
    
    for (int x = rangeMin.x; x <= rangeMax.x; x++)
    {
        for (int y = rangeMin.y; y <= rangeMax.y; y++)
        {
            for (int z = rangeMin.z; z <= rangeMax.z; z++)
            {
                int3 adjacentCoord = int3(voxelCoord) + int3(x, y, z) + direction;

                if (any(adjacentCoord < 0) || any(adjacentCoord >= int3(GridDimension)))
                {
                    emptyVoxels += 1;
                    continue;
                }
                
                uint2 hashedPositionIndex = FindHashedCompactedPositionIndex(uint3(adjacentCoord), GridDimension);
                emptyVoxels += 1 - hashedPositionIndex.y;
            }
        }
    }
    
    return emptyVoxels;
}

float4 GetAccumulatedDot(uint index)
{
    float accDotProduct = 0.0f;
    uint fragmentIndex = gVoxelIndicesCompactBuffer[index];
            
    float3 lastNormal = normalize(gFragmentBuffer[fragmentIndex].normal);
    fragmentIndex = gNextIndexBuffer[fragmentIndex];
            
    while (fragmentIndex != UINT_MAX)
    {
        float3 currentNormal = normalize(gFragmentBuffer[fragmentIndex].normal);
        accDotProduct += dot(lastNormal, currentNormal);
                
        fragmentIndex = gNextIndexBuffer[fragmentIndex];
        lastNormal = currentNormal;
    }
    
    return float4(lastNormal.x, lastNormal.y, lastNormal.z, accDotProduct);
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
    

    uint normalDirection = 0;

    float4 accumulatedDot = GetAccumulatedDot(hashIndex);
    float maxAccCos = 0.0f;
    
    for (uint i = 0; i < 6; i++)
    {
        float currentAccCos = accumulatedDot.w + dot(accumulatedDot.xyz, axisDirections[i]);
                    
        if (currentAccCos > maxAccCos)
        {
            maxAccCos = currentAccCos;
            normalDirection = i;
        }
    }
    
    
    gVoxelNormalDirectionBuffer[hashIndex] = float3(axisDirections[normalDirection]);
}


[numthreads(8, 8, 8)]
void CS(uint3 GridID : SV_GroupID, uint GroupThreadIndex : SV_GroupIndex, uint3 GroupThreadID : SV_GroupThreadID)
{   
    uint3 GridSize = ceil(TileDimension / (512.0f)); // 8 * 8 * 8;
    
    uint threadLinearIndex = GridID.z * (GridSize.x * GridSize.y) + GridID.y * GridSize.x + GridID.x;
    threadLinearIndex = threadLinearIndex * 512 + GroupThreadIndex;

    uint3 TileID = GridID * 8 + GroupThreadID;
    
    if (CurrentPhase == 0)
    {
        if (threadLinearIndex == 0)
        {
            gClusterCounterBuffer[0] = 0;
        }
            
        SetUpVoxelNormal(GridID.x * 512 + GroupThreadIndex);
    }
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

        
        uint tileClusterGrid = ceil(pow(numberOfClusterInTile, 1 / 3.0f));
        
        float clusterOffset = (2.0f * S) / tileClusterGrid;
        
        ClusterData data;
        data.VoxelCount = 0;
        data.Normal = float3(1.0f, 1.0f, 1.0f);
        data.FirstDataIndex = UINT_MAX;

        uint originalValue = 0;
        
        uint3 FirstVoxelInTile = TileID * 2 * S;
        
          
        
        for (uint nCluster = 0; nCluster < numberOfClusterInTile; nCluster++)
        {
            float3 pos = GetVoxelPosition(nCluster, uint3(tileClusterGrid, tileClusterGrid, tileClusterGrid));

            pos = FirstVoxelInTile + (pos + 0.5f) * clusterOffset;

            data.Center = pos;
            
            InterlockedAdd(gClusterCounterBuffer[0], 1, originalValue);
            
            if (originalValue >= K)
                return;
            
            gClusterDataBuffer[originalValue] = data;
            gNextCluster[originalValue] = UINT_MAX;
        }
        
        
    }
    else if (CurrentPhase == 2)
    {        
        uint NumberOfThreads = GridSize.x * GridSize.y * GridSize.z * 512;
        uint voxelsPerThread = ceil((float) VoxelCount / NumberOfThreads);
        uint clusterPerThread = ceil((float) K / (NumberOfThreads));
       
        uint initialindex = threadLinearIndex * voxelsPerThread;
        uint finalindex = min(initialindex + voxelsPerThread, VoxelCount);
        
        for (uint i = initialindex; i < finalindex; i++)
        {
            gClusterAssignmentBuffer[i] = UINT_MAX;
            gClusterDistanceBuffer[i] = UINT_MAX;
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
            
            while (voxelDataIndex != UINT_MAX)
            {

                
                posAverage += GetVoxelPosition(gVoxelHashedCompactBuffer[voxelDataIndex], GridDimension);
                normalAverage += gVoxelNormalDirectionBuffer[voxelDataIndex];
                
                nVoxels += 1;
                
                uint nextIndex = gNextVoxelClusterData[voxelDataIndex];
                gNextVoxelClusterData[voxelDataIndex] = UINT_MAX;
                voxelDataIndex = nextIndex;
            }
            
            if (FirstClusterSet > 0 && nVoxels > 0)
            {
                cData.Center = posAverage / nVoxels;
                // No need to divide since we are normalizing
                cData.Normal = normalize(normalAverage);
            }
            cData.VoxelCount = nVoxels;
            cData.FirstDataIndex = UINT_MAX;
            
            gClusterDataBuffer[j] = cData;
            
            uint3 tileCoord = (uint3) floor(cData.Center / (2 * S));
            
            uint prev = UINT_MAX;
            uint currentValue;
            InterlockedCompareExchange(gTileBuffer[tileCoord], prev, j, currentValue);
        
            [allow_uav_condition]
            while (currentValue != prev)
            {
                prev = currentValue;
                gNextCluster[j] = currentValue;
                InterlockedCompareExchange(gTileBuffer[tileCoord], prev, j, currentValue);
            }
        }

    }
    else if (CurrentPhase == 3)
    {
        // Only tiles which are not adjacent are updated (corner tiles are considered adjacent)
        // So we check if the tile is even or odd for each axis
        
        
        uint threadLinearIndex = GridID.x * 512 + GroupThreadIndex;
        
        if (threadLinearIndex >= VoxelCount)
            return;
        
        uint voxelLinearCoord = gVoxelHashedCompactBuffer[threadLinearIndex];
        
        uint3 voxelCoord = GetVoxelPosition(voxelLinearCoord, GridDimension);
        
        uint3 voxelTile = (uint3) floor(voxelCoord / (2 * S));
        
        // Returns -1 or 1 for each axis depending on which half of the tile the voxel is in
        int3 voxelHalfInTile = round((voxelCoord - (voxelTile * 2 * S)) / (2.0f * S)) * 2 - 1;
        
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
        float minDistance = gClusterDistanceBuffer[threadLinearIndex];
        uint closestClusterIndex = UINT_MAX;
        
        float3 avgNormal = gVoxelNormalDirectionBuffer[threadLinearIndex];
        
        for (uint t = 0; t < adjacentTileCount; t++)
        {
            uint clusterIndex = gTileBuffer[adjacentTiles[t]];
            
            while (clusterIndex != UINT_MAX)
            {
                ClusterData cData = gClusterDataBuffer[clusterIndex];
                
                float3 d = float3(abs(cData.Center.x - voxelCoord.x),
                          abs(cData.Center.y - voxelCoord.y),
                          abs(cData.Center.z - voxelCoord.z));

                if (any(d > 1.0f * S))
                {
                    clusterIndex = gNextCluster[clusterIndex];
                    continue;
                }
                
                float dotProduct = cData.VoxelCount > 0 ? dot(avgNormal, cData.Normal) : cos30 + EPSILON;
                
                float distance = fraction * (d.x + d.y + d.z) * (1.0f - dotProduct);
                
                if (dotProduct > cos30)
                {
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                        closestClusterIndex = clusterIndex;
                    }
                }
                else if (FirstClusterSet == 0 && clusterIndex == UINT_MAX)
                {
                    clusterIndex = gNextCluster[clusterIndex];
                }

                clusterIndex = gNextCluster[clusterIndex];
            }
        }
        
        gClusterDistanceBuffer[threadLinearIndex] = minDistance;
        gClusterAssignmentBuffer[threadLinearIndex] = closestClusterIndex;
        
        uint prev = UINT_MAX;
        uint currentValue;
        InterlockedCompareExchange(gClusterDataBuffer[closestClusterIndex].FirstDataIndex, prev, threadLinearIndex, currentValue);

                        
        [allow_uav_condition]
        while (currentValue != prev)
        {
            prev = currentValue;
            gNextVoxelClusterData[threadLinearIndex] = currentValue;
            InterlockedCompareExchange(gClusterDataBuffer[closestClusterIndex].FirstDataIndex, prev, threadLinearIndex, currentValue);
        }
             
    }
    else if (CurrentPhase == 4)
    {
        uint threadLinearIndex = GridID.x * 512 + GroupThreadIndex;
        
        uint clustersPerThread = ceil((float) K / VoxelCount * 512);
        
        if (threadLinearIndex >= VoxelCount)
            return;
        
        if (threadLinearIndex < TileDimension.x * TileDimension.y * TileDimension.z)
        {
            uint3 tileId = GetVoxelPosition(threadLinearIndex, TileDimension);
            gTileBuffer[tileId] = UINT_MAX;
        }

        if (threadLinearIndex >= K)
            return;
        
        for (uint i = threadLinearIndex * clustersPerThread; i < threadLinearIndex * clustersPerThread + clustersPerThread; i++)
        {
            gNextCluster[i] = UINT_MAX;
        }
    }
    else if (CurrentPhase == 5)
    {
        uint NumberOfThreads = GridSize.x * GridSize.y * GridSize.z * 512;
        uint voxelsPerThread = ceil((float) VoxelCount / NumberOfThreads);
        uint clusterPerThread = ceil((float) K / (NumberOfThreads));
       
        
        uint initialindex = threadLinearIndex * clusterPerThread;
        uint finalindex = min(initialindex + clusterPerThread, K);
        
        for (uint j = initialindex; j < finalindex; j++)
        {
            ClusterData cData = gClusterDataBuffer[j];
            
            uint voxelDataIndex = cData.FirstDataIndex;
            uint nVoxels = 0;
            float3 posAverage = float3(0.0f, 0.0f, 0.0f);
            float3 normalAverage = float3(0.0f, 0.0f, 0.0f);
            
            while (voxelDataIndex != UINT_MAX)
            {

                posAverage += GetVoxelPosition(gVoxelHashedCompactBuffer[voxelDataIndex], GridDimension);
                normalAverage += gVoxelNormalDirectionBuffer[voxelDataIndex];
                
                nVoxels += 1;
                
                uint nextIndex = gNextVoxelClusterData[voxelDataIndex];
                gNextVoxelClusterData[voxelDataIndex] = UINT_MAX;
                voxelDataIndex = nextIndex;
            }
            
            if (nVoxels < 1)
                return;
            


            cData.Center = posAverage / nVoxels;
            // No need to divide since we are normalizing
            cData.Normal = normalize(normalAverage);
            cData.VoxelCount = nVoxels;
  
            cData.FirstDataIndex = UINT_MAX;
            
            gClusterDataBuffer[j] = cData;
            
            InterlockedAdd(gClusterCounterBuffer[0], 1);
        }
    }
}