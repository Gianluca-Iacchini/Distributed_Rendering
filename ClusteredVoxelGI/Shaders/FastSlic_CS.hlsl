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

RWStructuredBuffer<ClusterData> gSubClusterData : register(u9, space0);



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
        voxelDataIndex = gNextVoxelClusterData[voxelDataIndex];
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

            pos = round(FirstVoxelInTile + S); //(pos + 0.5f) * clusterOffset;

            data.Center = clamp(pos, 0.0f, (float)GridDimension);
            
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
            float3 lastAvg = float3(0.0f, 0.0f, 0.0f);
            
            while (voxelDataIndex != UINT_MAX)
            {
                posAverage += GetVoxelPosition(gVoxelHashedCompactBuffer[voxelDataIndex], GridDimension);
                normalAverage += gVoxelNormalDirectionBuffer[voxelDataIndex];
                
                nVoxels += 1;
                
                if (!(all(normalAverage < EPSILON) && all(normalAverage > -EPSILON)))
                {
                    lastAvg = normalAverage;
                }
                
                uint nextIndex = gNextVoxelClusterData[voxelDataIndex];
                gNextVoxelClusterData[voxelDataIndex] = UINT_MAX;
                voxelDataIndex = nextIndex;

            }
                     
            if (FirstClusterSet > 0 && nVoxels > 0)
            {
                cData.Center = posAverage / nVoxels;
                // No need to divide since we are normalizing
                cData.Normal = normalize(lastAvg);
            }
            cData.VoxelCount = nVoxels;
            cData.FirstDataIndex = UINT_MAX;
            cData.Center = clamp(round(cData.Center), 0.0f, (float)GridDimension);
            
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
        float minDistance = UINT_MAX;// gClusterDistanceBuffer[threadLinearIndex];
        uint closestClusterIndex = UINT_MAX;
        
        bool noCloseAngle = true;
        float minSpatialDistance = UINT_MAX;
        
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
                
                float distance = fraction * (d.x + d.y + d.z);
                
                if (dotProduct > cos30)
                {                    
                    dotProduct = 6 * S * (1.0f - dotProduct);
                    distance = distance + dotProduct;
                    
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                        closestClusterIndex = clusterIndex;
                    }
                    
                    noCloseAngle = false;
                }
                //else if (FirstClusterSet == 0 && closestClusterIndex == UINT_MAX)
                //{
                //    closestClusterIndex = clusterIndex;
                //}
                else if (noCloseAngle && distance < minSpatialDistance)
                {

                    minSpatialDistance = distance;
                    closestClusterIndex = clusterIndex;
                    
                }
                else if (closestClusterIndex == UINT_MAX)
                {
                    closestClusterIndex = clusterIndex;
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
        
        if (threadLinearIndex == 0)
        {
            gClusterCounterBuffer[0] = 0;
        }
    }
    else if (CurrentPhase == 5)
    {
        if (threadLinearIndex >= K)
            return;
        
        ClusterData cData = gClusterDataBuffer[threadLinearIndex];
            
        uint voxelDataIndex = cData.FirstDataIndex;
        uint nVoxels = 0;
        float3 posAverage = float3(0.0f, 0.0f, 0.0f);
        float3 normalAverage = float3(0.0f, 0.0f, 0.0f);
            
        while (voxelDataIndex != UINT_MAX)
        {                
            nVoxels += 1;
                
            voxelDataIndex = gNextVoxelClusterData[voxelDataIndex];
        }
            
        if (nVoxels < 1)
            return;
            
        uint originalValue = 0;
        InterlockedAdd(gClusterCounterBuffer[0], 1, originalValue);

        voxelDataIndex = cData.FirstDataIndex;
        while (voxelDataIndex != UINT_MAX)
        {                
            posAverage += GetVoxelPosition(gVoxelHashedCompactBuffer[voxelDataIndex], GridDimension);
            
            gClusterAssignmentBuffer[voxelDataIndex] = originalValue;
            
            uint nextIndex = gNextVoxelClusterData[voxelDataIndex];
            gNextVoxelClusterData[voxelDataIndex] = UINT_MAX;
            voxelDataIndex = nextIndex;
        }
        
        float3 axisDirections[6] =         {
            float3(1, 0, 0),
            float3(-1, 0, 0),
            float3(0, 1, 0),
            float3(0, -1, 0),
            float3(0, 0, 1),
            float3(0, 0, -1)
        };
        
        float minSum = UINT_MAX;
        
        for (uint i = 0; i < 6; i++)
        {
            float currentAccCos = GetClusterNormalDirection(threadLinearIndex, axisDirections[i]);
                    
            if (currentAccCos < minSum)
            {
                minSum = currentAccCos;
                cData.Normal = axisDirections[i];
            }
        }
        
        cData.Center = clamp(round(posAverage / nVoxels), 0, (float)GridDimension);
        // No need to divide since we are normalizing
        cData.VoxelCount = nVoxels;
  
        cData.FirstDataIndex = UINT_MAX;
            
        gSubClusterData[originalValue] = cData;
        
        uint3 tileCoord = (uint3) floor(cData.Center / (2 * S));
            
        uint prev = UINT_MAX;
        uint currentValue;
        InterlockedCompareExchange(gTileBuffer[tileCoord], prev, originalValue, currentValue);
        
        [allow_uav_condition]
        while (currentValue != prev)
        {
            prev = currentValue;
            gNextCluster[originalValue] = currentValue;
            InterlockedCompareExchange(gTileBuffer[tileCoord], prev, originalValue, currentValue);
        }
        
    }
    else if (CurrentPhase == 6)
    {
        if (threadLinearIndex >= VoxelCount)
            return;
        
        uint3 voxelPos = GetVoxelPosition(gVoxelHashedCompactBuffer[threadLinearIndex], GridDimension);
        uint clusterIndex = gClusterAssignmentBuffer[threadLinearIndex];
        
        if (clusterIndex == UINT_MAX)
            return;
        
        ClusterData cData = gSubClusterData[clusterIndex];
        
        if (any(abs(voxelPos - cData.Center) > 1.0f * S))
        {
            gClusterAssignmentBuffer[threadLinearIndex] = UINT_MAX;
        }

    }
}