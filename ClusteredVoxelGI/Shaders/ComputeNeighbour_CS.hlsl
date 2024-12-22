#include "../../VoxelUtils/Shaders/VoxelUtils.hlsli"

ConstantBuffer<ConstantBufferVoxelCommons> cbVoxelCommons : register(b0);
ConstantBuffer<ConstantBufferComputeNeighbour> cbComputeNeighbour : register(b1);

RWStructuredBuffer<ClusterData> gClusterDataBuffer : register(u0, space0);

bool intervalIntersection(float minX0, float maxX0, float minX1, float maxX1)
{
    if ((((minX0 >= minX1) && (minX0 <= maxX1)) || ((maxX0 >= minX1) && (maxX0 <= maxX1))) ||
		(((minX1 >= minX0) && (minX1 <= maxX0)) || ((maxX1 >= minX0) && (maxX1 <= maxX0))))
    {
        return true;
    }

    return false;
}

[numthreads(256, 1, 1)]
void CS( uint3 DTid : SV_DispatchThreadID )
{
    
    uint threadStart = DTid.x * cbComputeNeighbour.ElementsPerThread;
    uint threadEnd = min(threadStart + cbComputeNeighbour.ElementsPerThread, cbComputeNeighbour.TotalComputationCount);
    
    for (uint i = threadStart; i < threadEnd; i++)
    {
        uint iIndex = (uint) floor( (-1.0f + sqrt(1.0f + 8.0f * i)) / 2.0f);
        uint jIndex = i - iIndex * (iIndex + 1) / 2;
        
        if (iIndex == jIndex)
            continue;

        ClusterData clusterDataI = gClusterDataBuffer[iIndex];
        ClusterData clusterDataJ = gClusterDataBuffer[jIndex];
        

        bool result0 = intervalIntersection(clusterDataI.MinAABB.x - 1, clusterDataI.MaxAABB.x + 1, clusterDataJ.MinAABB.x, clusterDataJ.MaxAABB.x);
        bool result1 = intervalIntersection(clusterDataI.MinAABB.y - 1, clusterDataI.MaxAABB.y + 1, clusterDataJ.MinAABB.y, clusterDataJ.MaxAABB.y);
        bool result2 = intervalIntersection(clusterDataI.MinAABB.z - 1, clusterDataI.MaxAABB.z + 1, clusterDataJ.MinAABB.z, clusterDataJ.MaxAABB.z);

        
        if (result0 && result1 && result2)
        {
            uint numberNeighboursI = 0;
            uint numberNeighboursJ = 0;
            
            InterlockedAdd(gClusterDataBuffer[iIndex].NeighbourCount, 1, numberNeighboursI);
            InterlockedAdd(gClusterDataBuffer[jIndex].NeighbourCount, 1, numberNeighboursJ);
            
            if (numberNeighboursI < 64)
            {
                gClusterDataBuffer[iIndex].ClusterNeighbours[numberNeighboursI] = jIndex;
            }
            if (numberNeighboursJ < 64)
            {
                gClusterDataBuffer[jIndex].ClusterNeighbours[numberNeighboursJ] = iIndex;
            }
        }
    }

}