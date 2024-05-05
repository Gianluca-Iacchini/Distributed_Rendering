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
