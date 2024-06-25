#include "VoxelUtils.hlsli"

VertexOut VS( VertexIn vIn )
{
    VertexOut vOut;
    
    float4 posW = mul(float4(vIn.PosL, 1.0f), oWorld);
    
    vOut.PosW = posW.xyz;
    vOut.NormalW = mul(vIn.NormalL, (float3x3) oWorld);
    vOut.PosH = posW;
    vOut.ShadowPosH = vOut.PosH;
    vOut.Tex = vIn.Tex;
    vOut.ProjAxis = 0;
    
    return vOut;
}