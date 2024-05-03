cbuffer Commons : register(b0)
{
    float4x4 view : packoffset(c0);
    float4x4 invView : packoffset(c4);
    float4x4 proj : packoffset(c8);
    float4x4 invProj : packoffset(c12);
    float4x4 viewProj : packoffset(c16);
    float4x4 invViewProj : packoffset(c20);
    float3 eyePos : packoffset(c24);
    float nearPlane : packoffset(c24.w);
    float2 renderTargetSize : packoffset(c25);
    float2 invRenderTargetSize : packoffset(c25.z);

    float farPlane : packoffset(c26);
    float totalTime : packoffset(c26.y);
    float deltaTime : packoffset(c26.z);
};

cbuffer Object : register(b1)
{
    float4x4 world : packoffset(c0);
    float4x4 invWorld : packoffset(c4);
    float4x4 texTransform : packoffset(c8);
    uint materialIndex : packoffset(c12);
};

Texture2D gTex[6] : register(t0);

SamplerState gSampler : register(s0);

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
    //float3 sinCos = float3(sin(totalTime), cos(totalTime), sin(totalTime)) * 0.5f;
    float4 posW = mul(float4(vIn.PosL, 1.0f), world);
    vOut.NormalW = mul(vIn.Normal, (float3x3)world);
    vOut.PosH = mul(posW, viewProj);
    vOut.Tex = vIn.Tex;
    
    return vOut;
}

float4 PS(VertexOut pIn) : SV_TARGET
{
    //return float4(normalize(pIn.NormalW), 1.0f);
    return gTex[0].Sample(gSampler, pIn.Tex);
}