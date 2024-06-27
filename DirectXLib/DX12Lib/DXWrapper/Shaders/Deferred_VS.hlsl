#include "Common.hlsli"

VertexOutPosTex VS(VertexInPosTex vertexIn)
{
    VertexOutPosTex vOut;
    vOut.PosH = float4(vertexIn.PosL, 1.0f);
    vOut.Tex = vertexIn.Tex;
    
    return vOut;
}