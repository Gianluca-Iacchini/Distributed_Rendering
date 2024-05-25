#include "Common.hlsli"

struct VSInput
{
    float3 Pos : SV_Position;
    float3 NORMAL : NORMAL;
    float2 Tex : TEXCOORD;
};

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float2 Tex : TEXCOORD;
};

VSOutput VS(VSInput input)
{
    float4 posW = mul(float4(input.Pos, 1.0f), oWorld);
    
    VSOutput output;
    
    output.Pos = mul(posW, cViewProj);
    output.Tex = input.Tex;
    return output;
}