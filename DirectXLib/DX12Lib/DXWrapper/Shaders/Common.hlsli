#include "LightingUtil.hlsli"

struct Commons
{
    float2 RenderTargetSize;
    float2 InvRenderTargetSize;

    float TotalTime;
    float DeltaTime;
    int NumLights;
    float _pad0;
    float _pad1;
    float _pad2;
};

struct Camera
{
    float4x4 View;
    float4x4 InvView;
    float4x4 Proj;
    float4x4 InvProj;
    float4x4 ViewProj;
    float4x4 InvViewProj;
    
    float3 EyePos;
    float NearPlane;
    
    float FarPlane;
    float _pad0;
    float _pad1;
    float _pad2;
};

struct Object
{
    float4x4 World;
    float4x4 InvWorld;
    float4x4 TexTransform;
    
    uint MaterialIndex;
    float _pad0;
    float _pad1;
    float _pad2;
};



Texture2D gShadowMap : register(t0);
RWTexture3D<float4> gVoxelGrid : register(u0);


SamplerState gSampler : register(s0);
SamplerComparisonState gShadowSampler : register(s1);

struct VertexInPosNormalTex
{
    float3 PosL : SV_Position;
    float3 NormalL : NORMAL;
    float2 Tex : TEXCOORD;
};

struct VertexOutPosNormalTex
{
    float4 PosH : SV_POSITION;
    float3 PosW : POSITION0;
    float3 NormalW : NORMAL;
    float2 Tex : TEXCOORD;
};

struct VertexInPosTex
{
    float3 PosL : SV_Position;
    float2 Tex : TEXCOORD;
};

struct VertexOutPosTex
{
    float4 PosH : SV_POSITION;
    float2 Tex : TEXCOORD;

};

float CalcShadowFactor(float4 shadowPosH)
{
    // Complete projection by doing division by w.
    shadowPosH.xyz /= shadowPosH.w;

    // Depth in NDC space.
    float depth = shadowPosH.z;

    uint width, height, numMips;
    gShadowMap.GetDimensions(0, width, height, numMips);

    // Texel size.
    float dx = 1.0f / (float) width;
    float dy = 1.0f / (float) height;

    float percentLit = 0.0f;

    // Use different offsets for different quality levels
    float2 offsets[9] =
    {
        float2(-1.0f * dx, -1.0f * dy), float2(0.0f * dx, -1.0f * dy), float2(1.0f * dx, -1.0f * dy),
        float2(-1.0f * dx, 0.0f * dy), float2(0.0f * dx, 0.0f * dy), float2(1.0f * dx, 0.0f * dy),
        float2(-1.0f * dx, 1.0f * dy), float2(0.0f * dx, 1.0f * dy), float2(1.0f * dx, 1.0f * dy)
    };

    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        float2 shadowPosOffset = shadowPosH.xy + offsets[i];

        // Hack to remove shadowmap edge artifacts
        if (shadowPosOffset.x <= 0.01f || shadowPosOffset.x >= 0.99f || shadowPosOffset.y <= 0.01f || shadowPosOffset.y >= 0.99f)
            continue;

        percentLit += gShadowMap.SampleCmpLevelZero(gShadowSampler,
        shadowPosOffset, depth).r;
    }

    return percentLit / 9.0f;
}