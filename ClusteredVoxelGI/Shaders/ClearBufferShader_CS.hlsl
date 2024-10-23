#include "VoxelUtils.hlsli"

ConstantBuffer<ConstantBufferClearBuffers> cbClearBuffers : register(b0);

RWStructuredBuffer<float3> gVoxelRadiance : register(u0);
RWStructuredBuffer<uint4> gClusterRadiance : register(u1);


[numthreads(128, 1, 1)]
void CS( uint3 DTid : SV_DispatchThreadID )
{
    if (DTid.x >= cbClearBuffers.ValueCount0)
        return;
    
    gVoxelRadiance[DTid.x] = float3(0.0f, 0.0f, 0.0f);
    
    if (DTid.x >= cbClearBuffers.ValueCount1)
        return;
    
    gClusterRadiance[DTid.x] = uint4(0, 0, 0, 0);
}