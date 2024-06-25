#include "LightingUtil.hlsli"

cbuffer Commons : register(b0)
{
    float2 cRenderTargetSize : packoffset(c0);
    float2 cInvRenderTargetSize : packoffset(c0.z);

    float cTotalTime : packoffset(c1);
    float cDeltaTime : packoffset(c1.y);
    int cNumLights : packoffset(c1.z);
    bool cUseShadows : packoffset(c1.w);
};

cbuffer Object : register(b1)
{
    float4x4 oWorld : packoffset(c0);
    float4x4 oInvWorld : packoffset(c4);
    float4x4 oTexTransform : packoffset(c8);
    uint oMaterialIndex : packoffset(c12);
};

cbuffer Camera : register(b2)
{
    float4x4 cView : packoffset(c0);
    float4x4 cInvView : packoffset(c4);
    float4x4 cProj : packoffset(c8);
    float4x4 cInvProj : packoffset(c12);
    float4x4 cViewProj : packoffset(c16);
    float4x4 cInvViewProj : packoffset(c20);
    float3 cEyePos : packoffset(c24);
    float cNearPlane : packoffset(c24.w);
    float cFarPlane : packoffset(c25);
}

Texture2D gShadowMap : register(t0);
RWTexture3D<float4> gVoxelGrid : register(u0);

Texture2D gEmissiveTex : register(t1);
Texture2D gNormalMap : register(t2);
Texture2D gDiffuseTex : register(t3);
#ifndef PBR
Texture2D gSpecularTex : register(t4); 
Texture2D gAmbientTex : register(t5); 
Texture2D gShininessTex : register(t6);
Texture2D gOpacity : register(t7);
Texture2D gBumpMap : register(t8);
#else
Texture2D gMetallicRoughness : register(t4);
Texture2D gOcclusion : register(t5);
#endif


StructuredBuffer<Light> gLights : register(t0, space1);
StructuredBuffer<GenericMaterial> gMaterials : register(t1, space1);

SamplerState gSampler : register(s0);
SamplerComparisonState gShadowSampler : register(s1);

struct VertexIn
{
    float3 PosL : SV_Position;
    float3 NormalL : NORMAL;
    float2 Tex : TEXCOORD;
};

struct VertexOut
{
    float4 PosH : SV_POSITION;
    float4 ShadowPosH : POSITION0;
    float3 PosW : POSITION1;
    float3 NormalW : NORMAL;
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