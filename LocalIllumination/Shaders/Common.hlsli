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

cbuffer Material : register(b2)
{
    float4 diffuse : packoffset(c0);
    float4 specular : packoffset(c4);
    float4 ambient : packoffset(c8);
    float4 emissive : packoffset(c12);
    
    float opacity : packoffset(c16);
    float shininess : packoffset(c16.y);
    float refractiveIndex : packoffset(c16.z);
    float bumpIntensity : packoffset(c16.w);
};

Texture2D diffuseTex : register(t0);
Texture2D specularTex : register(t1);
Texture2D ambientTex : register(t2);
Texture2D emissiveTex : register(t3);
Texture2D normalMap : register(t4);
Texture2D bumpMap : register(t5);


SamplerState gSampler : register(s0);

/* https://github.com/microsoft/DirectXTK12/blob/main/Src/Shaders/Utilities.fxh */
// Christian Schuler, "Normal Mapping without Precomputed Tangents", ShaderX 5, Chapter 2.6, pp. 131-140
// See also follow-up blog post: http://www.thetenthplanet.de/archives/1180
float3x3 CalculateTBN(float3 p, float3 n, float2 tex)
{
    float3 dp1 = ddx(p);
    float3 dp2 = ddy(p);
    float2 duv1 = ddx(tex);
    float2 duv2 = ddy(tex);

    float3x3 M = float3x3(dp1, dp2, cross(dp1, dp2));
    float2x3 inverseM = float2x3(cross(M[1], M[2]), cross(M[2], M[0]));
    float3 t = normalize(mul(float2(duv1.x, duv2.x), inverseM));
    float3 b = normalize(mul(float2(duv1.y, duv2.y), inverseM));
    return float3x3(t, b, n);
}

float3 ComputeTwoChannelNormal(float2 normal)
{
    // Change normal mapping from [0, 1] to [-1, 1]
    float2 xy = 2.0f * normal - 1.0f;
    
    // Compute z from x and y
    float z = sqrt(1.0f - dot(xy, xy));

    return float3(xy.x, xy.y, z);

}