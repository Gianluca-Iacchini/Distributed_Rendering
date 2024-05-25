#include "Common.hlsli"


VertexOut VS(VertexIn vIn)
{   
    VertexOut vOut;
    
    float4 posW = mul(float4(vIn.PosL, 1.0f), oWorld);
    
    vOut.PosW = posW.xyz;
    vOut.NormalW = mul(vIn.NormalL, (float3x3)oWorld);
    vOut.PosH = mul(posW, cViewProj);
    vOut.ShadowPosH = mul(posW, gLights[0].shadowMatrix);

    vOut.Tex = vIn.Tex;
    
    return vOut;
}

