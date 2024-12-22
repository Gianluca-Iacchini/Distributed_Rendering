#include "../../VoxelUtils/Shaders/VoxelUtils.hlsli"

cbuffer cbPerObject : register(b2)
{
    Object object;
};

VertexOutVoxel VS( VertexInPosNormalTex vIn )
{
    VertexOutVoxel vOut;
    
    float4 posW = mul(float4(vIn.PosL, 1.0f), object.World);
    
    vOut.PosW = posW.xyz;
    vOut.NormalW = mul(vIn.NormalL, (float3x3) object.World);
    vOut.PosH = posW;
    vOut.Tex = vIn.Tex;
    vOut.ProjAxis = 0;
    
    return vOut;
}