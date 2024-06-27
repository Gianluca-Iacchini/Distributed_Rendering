#include "Common.hlsli"

cbuffer cbCamera : register(b1)
{
    Camera camera;
}

cbuffer cbPerObject : register(b2)
{
    Object object;
}


VertexOutPosNormalTex VS(VertexInPosNormalTex vIn)
{   
    VertexOutPosNormalTex vOut;
    
    float4 posW = mul(float4(vIn.PosL, 1.0f), object.World);
    
    vOut.PosW = posW.xyz;
    vOut.NormalW = mul(vIn.NormalL, (float3x3)object.World);
    vOut.PosH = mul(posW, camera.ViewProj);

    vOut.Tex = vIn.Tex;
    
    return vOut;
}

