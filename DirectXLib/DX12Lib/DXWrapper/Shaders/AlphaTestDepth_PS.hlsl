#include "Common.hlsli"

Texture2D gDiffuseTex : register(t3);


void PS(VertexOutPosTex vOut)
{
    float4 color = gDiffuseTex.Sample(gSampler, vOut.Tex);
    if (color.a < 0.1f)
        discard;
}