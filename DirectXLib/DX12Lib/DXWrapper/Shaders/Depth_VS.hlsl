#include "Common.hlsli"

cbuffer cbCamera : register(b1)
{
    Camera camera;
};

cbuffer cbPerObject : register(b2)
{
    Object object;
};



VertexOutPosTex VS(VertexInPosNormalTex input)
{
    float4 posW = mul(float4(input.PosL, 1.0f), object.World);
    
    VertexOutPosTex output;
    
    output.PosH = mul(posW, camera.ViewProj);
    output.Tex = input.Tex;
    return output;
}