#include "VoxelUtils.hlsli"

ConstantBuffer<ConstantBufferClearBuffers> cbClearBuffers : register(b0);

RWByteAddressBuffer gVoxelLitBuffer : register(u0);
RWStructuredBuffer<uint4> gClusterRadiance : register(u1);


[numthreads(128, 1, 1)]
void CS( uint3 DTid : SV_DispatchThreadID )
{
    // ValueCount = Voxel Count
    if (DTid.x >= cbClearBuffers.ValueCount0)
        return;
    
    uint idx = DTid.x >> 5u;
    idx = idx * 4;
    
    gVoxelLitBuffer.Store(idx, 0);

    // ValueCount1 = Cluster Count
    if (DTid.x >= cbClearBuffers.ValueCount1)
        return;
    
    gClusterRadiance[DTid.x] = uint4(0, 0, 0, 0);
}