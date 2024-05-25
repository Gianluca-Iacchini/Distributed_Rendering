#include "Common.hlsli"

struct VSOutput
{
    float4 Pos : SV_Position;
    float2 Tex : TEXCOORD;
};

void PS(VSOutput vOut)
{
    float4 color = gDiffuseTex.Sample(gSampler, vOut.Tex);
    if (color.a < 0.1f)
        discard;
}