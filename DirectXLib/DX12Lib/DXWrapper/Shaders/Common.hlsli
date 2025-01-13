#ifndef COMMON_HLSLI
#define COMMON_HLSLI

#include "LightingUtil.hlsli"

struct Commons
{
    float2 RenderTargetSize;
    float2 InvRenderTargetSize;

    float TotalTime;
    float DeltaTime;
    int NumLights;
    uint UseRTGI;
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

float CalcShadowFactor(Texture2D shadowMap, float4 shadowPosH)
{
    // Complete projection by doing division by w.
    shadowPosH.xyz /= shadowPosH.w;

    // Depth in NDC space.
    float depth = shadowPosH.z;

    uint width, height, numMips;
    shadowMap.GetDimensions(0, width, height, numMips);

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

        percentLit += shadowMap.SampleCmpLevelZero(gShadowSampler,
        shadowPosOffset, depth).r;
    }

    return percentLit / 9.0f;
}

float2 UnpackFloats16(uint packedUint)
{
    uint packedX = packedUint.x & 0xFFFF;
    uint packedY = ((packedUint.x >> 16) & 0xFFFF);
    
    return float2(f16tof32(packedX), f16tof32(packedY));
}

uint PackFloats16(float2 floatsToPack)
{
    uint radX16 = f32tof16(floatsToPack.x);
    uint radY16 = f32tof16(floatsToPack.y);
        
    uint packedRadX = (radY16 << 16) | radX16;
    
    return packedRadX;
}

uint PackFloat3ToUint(float3 color)
{
    // Clamp the color values to [0, 1] and scale to the appropriate range
    uint r = (uint) (clamp(color.r, 0.0, 1.0) * 1023.0); // 10 bits (0-1023)
    // Human eye is more sensitive to green, so we give it more bits
    uint g = (uint) (clamp(color.g, 0.0, 1.0) * 4095.0); // 12 bits (0-4095)
    uint b = (uint) (clamp(color.b, 0.0, 1.0) * 1023.0); // 10 bits (0-1023)

    // Pack the components into a single uint
    return (r << 22) | (g << 10) | b;
}

float3 UnpackUintToFloat3(uint packedColor)
{
    float r = float((packedColor >> 22) & 0x3FF) / 1023.0; // Extract 10 bits and normalize
    float g = float((packedColor >> 10) & 0xFFF) / 4095.0; // Extract 12 bits and normalize
    float b = float(packedColor & 0x3FF) / 1023.0;         // Extract 10 bits and normalize

    return float3(r, g, b);
}
#endif
