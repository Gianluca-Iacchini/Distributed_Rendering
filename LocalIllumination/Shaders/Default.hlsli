#include "Common.hlsli"

struct VertexIn
{
    float3 PosL : SV_Position;
    float3 Normal : NORMAL;
    float2 Tex : TEXCOORD;
};

struct VertexOut
{
    float4 PosH : SV_POSITION;
    float3 NormalW : NORMAL;
    float2 Tex : TEXCOORD;
};

VertexOut VS(VertexIn vIn)
{
    VertexOut vOut;
    float4 posW = mul(float4(vIn.PosL, 1.0f), world);
    vOut.NormalW = mul(vIn.Normal, (float3x3) world);
    vOut.PosH = mul(posW, viewProj);
    vOut.Tex = vIn.Tex;
    
    return vOut;
}

float4 PS(VertexOut pIn) : SV_TARGET
{
    //return float4(normalize(pIn.NormalW), 1.0f);
    return gTex[0].Sample(gSampler, pIn.Tex);
}