#include "LightingUtil.hlsli"

cbuffer Commons : register(b0)
{
    float4x4 cView : packoffset(c0);
    float4x4 cInvView : packoffset(c4);
    float4x4 cProj : packoffset(c8);
    float4x4 cInvProj : packoffset(c12);
    float4x4 cViewProj : packoffset(c16);
    float4x4 cInvViewProj : packoffset(c20);
    float3 cEyePos : packoffset(c24);
    float cNearPlane : packoffset(c24.w);
    float2 cRenderTargetSize : packoffset(c25);
    float2 cInvRenderTargetSize : packoffset(c25.z);

    float cFarPlane : packoffset(c26);
    float cTotalTime : packoffset(c26.y);
    float cDeltaTime : packoffset(c26.z);
    
    Light cDirLight : packoffset(c27);
};

cbuffer Object : register(b1)
{
    float4x4 oWorld : packoffset(c0);
    float4x4 oInvWorld : packoffset(c4);
    float4x4 oTexTransform : packoffset(c8);
    uint oMaterialIndex : packoffset(c12);
};

ConstantBuffer<Material> gMaterial : register(b2);



Texture2D gEmissiveTex : register(t0);
Texture2D gNormalMap : register(t1);
Texture2D gDiffuseTex : register(t2);
#ifndef PBR
Texture2D gSpecularTex : register(t3); 
Texture2D gAmbientTex : register(t4); 
Texture2D gShininessTex : register(t5);
Texture2D gBumpMap : register(t6);
#else
Texture2D gMetallicRoughness : register(t3);
Texture2D gOcclusion : register(t4);
#endif


StructuredBuffer<Material> gMaterials : register(t0, space1);

SamplerState gSampler : register(s0);

struct VertexIn
{
    float3 PosL : SV_Position;
    float3 NormalL : NORMAL;
    float2 Tex : TEXCOORD;
};

struct VertexOut
{
    float4 PosH : SV_POSITION;
    float3 PosW : POSITION;
    float3 NormalW : NORMAL;
    float2 Tex : TEXCOORD;
};