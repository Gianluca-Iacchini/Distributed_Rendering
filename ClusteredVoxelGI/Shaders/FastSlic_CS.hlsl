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


ByteAddressBuffer gVoxelOccupiedBuffer : register(t0, space0);
StructuredBuffer<FragmentData> gFragmentBuffer : register(t1, space0);
StructuredBuffer<uint> gNextIndexBuffer : register(t2, space0);

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space1);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space1);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space1);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space1);


// Buffer containing the cluster data used by this shader
// Dim: K
RWStructuredBuffer<ClusterData> gClusterDataBuffer : register(u0, space0);

// Buffer containing voxels that are assigned to a cluster in a contiguous way.
// Voxels belonging to the same clusters are stored in a sequence between FirstDataIndex and FirstDataIndex + VoxelCount.
RWStructuredBuffer<uint> gVoxelInClusterBuffer : register(u1, space0);

// Buffer representing a mapping for voxels assigned to a cluster. This represents the same voxel assignment as the previous buffer,
// But while the previous one is a linked list, this one is a direct mapping. The previous one is useful for retrieving all the voxels
// assigned for a given cluster idx, this one is useful for retrieving the cluster idx for a given voxel idx.
// e.g. gVoxelAssigmnetMap[2] = 1002 means that voxel 2 is assigned to cluster 1002
// Dim: VoxelCount
RWStructuredBuffer<uint> gVoxelAssignmentMap : register(u2, space0);

// Buffer containing the color of each voxel, obtained as the average of the color of the fragments in the voxel
RWStructuredBuffer<float3> gVoxelColorBuffer : register(u3, space0);

// Buffer containing the normal direction of each voxel. All the normal directions are set to one of the axis positive or negative directions
// Depending on the nearby empty voxels and the normal direction of the fragments in the voxel
// Dim: VoxelCount
RWStructuredBuffer<float3> gVoxelNormalDirectionBuffer : register(u4, space0);

// 3D texture representing the subdivision of the space. Each element (Tile) represents a size of the space equal to (2S x 2S x 2S)
// Dim: (GridDimension.x / 2S) x (GridDimension.y / 2S) x (GridDimension.z / 2S)
RWTexture3D<uint> gTileBuffer : register(u5, space0);

// Buffer representing the linked list of clusters in each tile. Each element is the index of the next cluster in the same tile.
// The first cluster is stored in the TileBuffer.
// e.g. gTileBuffer[0,0,0] = 10, gNextClusterInTileLinkedList[10] = 21, gNextClusterInTileLinkedList[21] = UINT_MAX
// means that at the tile (0,0,0) we have clusters 10 and 21.
// Dim: K
RWStructuredBuffer<uint> gNextClusterInTileLinkedList : register(u6, space0);


// Generic counter buffer
// Dim: 1
RWStructuredBuffer<uint> gCounter : register(u7, space0);

// Buffer used to store cluster data, used as a temporary buffer for multi threaded operations
// Will also be used on the next step for cluster merging
RWStructuredBuffer<ClusterData> gSubclusterDataBuffer : register(u8, space0);

// Buffer representing the final voxel assignment for each cluster, as a linked list
// The first element is stored in the field FirstDataIndex in the ClusterData struct, and
// each element is the index of the next element in the list 
// e.g. gClusterDataBuffer[1002].FirstDataIndex = 2, gNextVoxelLinkedList[2] = 2013 gNextVoxelLinkedList[2013] = UINT_MAX
// means that cluster 1002 has voxels 2 and 2013 assigned to it.
// Dim: VoxelCount
RWStructuredBuffer<uint> gNextVoxelLinkedList : register(u9, space0);

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
    bool isInBoundsPos = IsWithinBounds(voxelCoord, int3(1, 0, 0), GridDimension);
    bool isInBoundsNeg = IsWithinBounds(voxelCoord, int3(-1, 0, 0), GridDimension);
    
    int3 negNeighbour = int3(voxelCoord.x - 1, voxelCoord.y, voxelCoord.z);
    int3 posNeighbour = int3(voxelCoord.x + 1, voxelCoord.y, voxelCoord.z);
    
    uint emptyVoxels = 0;

    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            int3 adjacentOffset = int3(0, i, j);
            bool isNeighbourInBounds = false;
            if (isInBoundsNeg)
            {
                isNeighbourInBounds = IsWithinBounds(negNeighbour, adjacentOffset, GridDimension);
                if (isNeighbourInBounds)
                {
                    emptyVoxels += 1 - (uint) IsVoxelPresent(uint3(negNeighbour + adjacentOffset), GridDimension, gVoxelOccupiedBuffer);
                }
            }

            if (isInBoundsPos)
            {
                isNeighbourInBounds = IsWithinBounds(posNeighbour, adjacentOffset, GridDimension);
                if (isNeighbourInBounds)
                {
                    emptyVoxels += 1 - (uint) IsVoxelPresent(uint3(posNeighbour + adjacentOffset), GridDimension, gVoxelOccupiedBuffer);
                }
            }
        }
    }
    
    return emptyVoxels;
}

uint GetEmptyVoxelY(uint3 voxelCoord)
{
    bool isInBoundsPos = IsWithinBounds(voxelCoord, int3(0, 1, 0), GridDimension);
    bool isInBoundsNeg = IsWithinBounds(voxelCoord, int3(0, -1, 0), GridDimension);
    
    int3 negNeighbour = int3(voxelCoord.x, voxelCoord.y - 1, voxelCoord.z);
    int3 posNeighbour = int3(voxelCoord.x, voxelCoord.y + 1, voxelCoord.z);
    
    uint emptyVoxels = 0;

    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            int3 adjacentOffset = int3(i, 0, j);
            bool isNeighbourInBounds = false;
            if (isInBoundsNeg)
            {
                isNeighbourInBounds = IsWithinBounds(negNeighbour, adjacentOffset, GridDimension);
                if (isNeighbourInBounds)
                {
                    emptyVoxels += 1 - (uint) IsVoxelPresent(uint3(negNeighbour + adjacentOffset), GridDimension, gVoxelOccupiedBuffer);
                }
            }

            if (isInBoundsPos)
            {
                isNeighbourInBounds = IsWithinBounds(posNeighbour, adjacentOffset, GridDimension);
                if (isNeighbourInBounds)
                {
                    emptyVoxels += 1 - (uint) IsVoxelPresent(uint3(posNeighbour + adjacentOffset), GridDimension, gVoxelOccupiedBuffer);
                }
            }
        }
    }
    
    return emptyVoxels;
}

uint GetEmptyVoxelZ(uint3 voxelCoord)
{
    bool isInBoundsPos = IsWithinBounds(voxelCoord, int3(0, 0, 1), GridDimension);
    bool isInBoundsNeg = IsWithinBounds(voxelCoord, int3(0, 0, -1), GridDimension);
    
    int3 negNeighbour = int3(voxelCoord.x, voxelCoord.y, voxelCoord.z - 1);
    int3 posNeighbour = int3(voxelCoord.x, voxelCoord.y, voxelCoord.z + 1);
    
    uint emptyVoxels = 0;

    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            int3 adjacentOffset = int3(i, j, 0);
            bool isNeighbourInBounds = false;
            if (isInBoundsNeg)
            {
                isNeighbourInBounds = IsWithinBounds(negNeighbour, adjacentOffset, GridDimension);
                if (isNeighbourInBounds)
                {
                    emptyVoxels += 1 - (uint) IsVoxelPresent(uint3(negNeighbour + adjacentOffset), GridDimension, gVoxelOccupiedBuffer);
                }
            }

            if (isInBoundsPos)
            {
                isNeighbourInBounds = IsWithinBounds(posNeighbour, adjacentOffset, GridDimension);
                if (isNeighbourInBounds)
                {
                    emptyVoxels += 1 - (uint) IsVoxelPresent(uint3(posNeighbour + adjacentOffset), GridDimension, gVoxelOccupiedBuffer);
                }
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

float3 GetVoxelAverageColor(uint voxelIdx)
{
    float3 sum = 0.0f;
    uint fragmentIndex = gVoxelIndicesCompactBuffer[voxelIdx];
    uint nFragments = 0;
    
    while (fragmentIndex != UINT_MAX)
    {
        sum += gFragmentBuffer[fragmentIndex].color.xyz;
        fragmentIndex = gNextIndexBuffer[fragmentIndex];
        nFragments += 1;    
    }
    
    sum = sum / nFragments;
    
    return sum;
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

        
        if (threadLinearIndex < TileDimension.x * TileDimension.y * TileDimension.z)
        {
            uint3 tileId = GetVoxelPosition(threadLinearIndex, TileDimension);
            gTileBuffer[tileId] = UINT_MAX;
        }
        
        // Precompute normal direction
        SetUpVoxelNormal(threadLinearIndex);
        gVoxelColorBuffer[threadLinearIndex] = GetVoxelAverageColor(threadLinearIndex);
        
        // Set the cluster assignment to UINT_MAX, which means no assignment
        gNextVoxelLinkedList[threadLinearIndex] = UINT_MAX;
        gVoxelAssignmentMap[threadLinearIndex] = UINT_MAX;
        
        if (threadLinearIndex < K)
        {
            gSubclusterDataBuffer[threadLinearIndex].FirstDataIndex = UINT_MAX;
            gNextClusterInTileLinkedList[threadLinearIndex] = UINT_MAX;
        }
    }
    else if (CurrentPhase == 1)
    {
        if (threadLinearIndex >= K)
            return;
        
        uint voxelsPerCluster = VoxelCount / K;
        
        uint voxelIdx = threadLinearIndex * voxelsPerCluster;
        

        voxelIdx = min(voxelIdx, VoxelCount - 1);
        
        uint originalValue = 0;
        InterlockedCompareExchange(gVoxelAssignmentMap[voxelIdx], UINT_MAX, threadLinearIndex, originalValue);
        
        if (originalValue == UINT_MAX)
        {
            ClusterData cData;
            cData.Center = GetVoxelPosition(gVoxelHashedCompactBuffer[voxelIdx], GridDimension);
            cData.MinAABB = cData.Center - 0.5f;
            cData.MaxAABB = cData.Center + 0.5f;
            cData.NeighbourCount = 0;
            cData.pad0 = 0.0f;
            cData.Normal = gVoxelNormalDirectionBuffer[voxelIdx];
            cData.VoxelCount = 1;
            cData.FirstDataIndex = voxelIdx;
            
            // Will be set later, there is no need to access the color buffer here.
            cData.Color = float3(0.0f, 0.0f, 0.0f);
            cData.FragmentCount = 0;

            [unroll]
            for (uint i = 0; i < 64; i++)
            {
                cData.ClusterNeighbours[i] = UINT_MAX;
            }
            
            gClusterDataBuffer[threadLinearIndex] = cData;
        }
    }
    // Called with K / 512, 1, 1 thread groups
    else if (CurrentPhase == 2)
    {

        uint NumberOfThreads = ceil(K / 512.0f) * 512;
        
        uint voxelsPerThread = ceil((float) VoxelCount / NumberOfThreads);
       
        uint initialindex = threadLinearIndex * voxelsPerThread;
        uint finalindex = min(initialindex + voxelsPerThread, VoxelCount);
            
        for (uint i = initialindex; i < finalindex; i++)
        {
            gVoxelAssignmentMap[i] = UINT_MAX;
        }
        

        
        if (threadLinearIndex >= K)
            return;
        

        ClusterData cData = gClusterDataBuffer[threadLinearIndex];
            
        uint voxelDataIndex = cData.FirstDataIndex;
        uint nVoxels = 0;
        float3 posAverage = float3(0.0f, 0.0f, 0.0f);
        float3 normalAverage = float3(0.0f, 0.0f, 0.0f);
        float3 colorAverage = float3(0.0f, 0.0f, 0.0f);
        uint3 minAABB = GridDimension + 10;
        uint3 maxAABB = uint3(0, 0, 0);
        float nFragments = 0;
                    
            
        while (voxelDataIndex != UINT_MAX)
        {
            uint3 voxelPos = GetVoxelPosition(gVoxelHashedCompactBuffer[voxelDataIndex], GridDimension);
            posAverage += float3(voxelPos);
            normalAverage += gVoxelNormalDirectionBuffer[voxelDataIndex];
            colorAverage += gVoxelColorBuffer[voxelDataIndex];
            
            minAABB = min(minAABB, voxelPos);
            maxAABB = max(maxAABB, voxelPos);
            
            nVoxels++;
                
            uint nextIdx = gNextVoxelLinkedList[voxelDataIndex];
            gNextVoxelLinkedList[voxelDataIndex] = UnassignedOnlyPass == 1 ? nextIdx : UINT_MAX;
            voxelDataIndex = nextIdx;
        }
            
                     
        if (nVoxels > 0)
        {
            posAverage = floor(posAverage / nVoxels);
            posAverage = clamp(posAverage, 0.0f, (float) GridDimension);
            cData.Center = uint3(posAverage);
            cData.Color = colorAverage / nVoxels;
            cData.MinAABB = float3(minAABB);
            cData.MaxAABB = float3(maxAABB);
            
            if (any(normalAverage > EPSILON) || any(normalAverage < -EPSILON))
            {
                cData.Normal = normalize(normalAverage);
            }
        }
        
        cData.VoxelCount = nVoxels;
        
        if (UnassignedOnlyPass == 0)
            cData.FirstDataIndex = UINT_MAX;

            
        gClusterDataBuffer[threadLinearIndex] = cData;
            
        uint3 tileCoord = (uint3) floor(cData.Center / (2 * S));
            
        
        uint prev = UINT_MAX;
        uint currentValue;
        InterlockedCompareExchange(gTileBuffer[tileCoord], prev, threadLinearIndex, currentValue);
        
        [allow_uav_condition]
        while (currentValue != prev)
        {
            prev = currentValue;
            gNextClusterInTileLinkedList[threadLinearIndex] = currentValue;
            InterlockedCompareExchange(gTileBuffer[tileCoord], prev, threadLinearIndex, currentValue);
        }
        

    }
    // Called with (VoxelCount / 512, 1, 1) thread groups
    else if (CurrentPhase == 3)
    {
        // Only tiles which are not adjacent are updated (corner tiles are considered adjacent)
        // So we check if the tile is even or odd for each axis
         
        if (threadLinearIndex >= VoxelCount)
            return;
        

        
        uint3 voxelCoord = GetVoxelPosition(gVoxelHashedCompactBuffer[threadLinearIndex], GridDimension);
        

        uint closestClusterIndex = UINT_MAX;
        
        int3 voxelTile = (int3) floor(voxelCoord / (2 * S));
        
        float3 voxelNormal = gVoxelNormalDirectionBuffer[threadLinearIndex];
        
        bool done = false;
        
        for (int i = -1; i <= 1 && !done; i++)
        {
            for (int j = -1; j <= 1 && !done; j++)
            {
                for (int k = -1; k <= 1 && !done; k++)
                {
                    if (!IsWithinBounds(voxelTile, int3(i, j, k), TileDimension))
                        continue;
                    
                    int3 tileCoord = voxelTile + int3(i, j, k);

                    uint clusterIndex = gTileBuffer[uint3(tileCoord)];
                    
                    while (clusterIndex != UINT_MAX && !done)
                    {
                        ClusterData cData = gClusterDataBuffer[clusterIndex];
                        
                        int3 distance = int3(cData.Center);
                        distance = abs(distance - int3(voxelCoord));
                        
                        if (all(distance <= int(S)))
                        {
                            if (dot(cData.Normal, voxelNormal) > cos25)
                            {
                                closestClusterIndex = clusterIndex;
                                done = true;
                            }
                        }
                        
                        clusterIndex = gNextClusterInTileLinkedList[clusterIndex];
                    }

                }

            }

        }
        
        
        gVoxelAssignmentMap[threadLinearIndex] = closestClusterIndex;
        
        if (closestClusterIndex == UINT_MAX)
            return;
        
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
        
        if (cData.VoxelCount > 0)
        {
            uint voxDataIdx = cData.FirstDataIndex;
            
            uint voxelStartIdx;
            InterlockedAdd(gCounter[1], cData.VoxelCount, voxelStartIdx);
            
            cData.FirstDataIndex = voxelStartIdx;
            
            uint subClusterIdx = 0;
            InterlockedAdd(gCounter[0], 1, subClusterIdx);
            gSubclusterDataBuffer[subClusterIdx] = cData;
            
            uint nVoxel = 0;
            while (voxDataIdx != UINT_MAX)
            {
                gVoxelInClusterBuffer[voxelStartIdx + nVoxel] = voxDataIdx;
                gVoxelAssignmentMap[voxDataIdx] = subClusterIdx;
                
                voxDataIdx = gNextVoxelLinkedList[voxDataIdx];
                nVoxel++;
            }
            

            
        }
        
        //if (cData.VoxelCount > 0)
        //{
        //    uint originalValue = 0;
        //    InterlockedAdd(gCounter[0], 1, originalValue);
        //    gSubclusterDataBuffer[originalValue] = cData;
            
        //    uint voxDataIdx = cData.FirstDataIndex;
            
        //    while (voxDataIdx != UINT_MAX)
        //    {
        //        gVoxelAssignmentMap[voxDataIdx] = originalValue;
        //        voxDataIdx = gNextVoxelLinkedList[voxDataIdx];
        //    }
        //}
        
        cData.Center = uint3(0, 0, 0);
        cData.Normal = float3(0.0f, 0.0f, 0.0f);
        cData.FirstDataIndex = UINT_MAX;
        cData.VoxelCount = 0;
        cData.Color = float3(0.0f, 0.0f, 0.0f);
        cData.FragmentCount = 0;
        cData.MinAABB = uint3(0, 0, 0);
        cData.MaxAABB = uint3(0, 0, 0);
        cData.NeighbourCount = 0;
        
        gClusterDataBuffer[threadLinearIndex] = cData;

    }

}