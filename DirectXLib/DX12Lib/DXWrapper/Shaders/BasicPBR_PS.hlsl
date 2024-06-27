#define PBR 1
#include "LightingUtil.hlsli"

struct VertexOut
{
    float4 PosH : SV_Position;
    float2 Tex : TEXCOORD;
};

cbuffer Commons : register(b0)
{
    float2 cRenderTargetSize : packoffset(c0);
    float2 cInvRenderTargetSize : packoffset(c0.z);

    float cTotalTime : packoffset(c1);
    float cDeltaTime : packoffset(c1.y);
    int cNumLights : packoffset(c1.z);
};

cbuffer Camera : register(b1)
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

Texture2D gBufferaWorld : register(t1);
Texture2D gBufferNormal : register(t2);
Texture2D gBufferDiffuse : register(t3);
Texture2D gBufferMetallicRoughnessAO : register(t4);

StructuredBuffer<Light> gLights : register(t0, space1);
StructuredBuffer<GenericMaterial> gMaterials : register(t1, space1);

SamplerState gSampler : register(s0);
SamplerComparisonState gShadowSampler : register(s1);

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


float4 PS(VertexOut pIn) : SV_Target
{
    

    
    float4 diffuse = gBufferDiffuse.Sample(gSampler, pIn.Tex);
    float4 worldPos = gBufferaWorld.Sample(gSampler, pIn.Tex);
    
    float2 normalPacked = gBufferNormal.Sample(gSampler, pIn.Tex).rg;
    float3 normal = UnpackNormal(normalPacked);
    float4 RMA = gBufferMetallicRoughnessAO.Sample(gSampler, pIn.Tex);
        
    // w coordinate is used to keep track of geometry. If there is no geometry, the w value is 1.0f so we can discard the pixel
    if (RMA.w >= 1.0f)
        discard;
    
#ifdef ALPHA_TEST
    if (diffuse.a < 0.1f)
    {
        discard;
    }
#endif
   
    
    PBRMaterial material = GetPBRMaterial(gMaterials[worldPos.w]);
    
    float3 V = normalize(cEyePos - worldPos.xyz);
    

    diffuse = diffuse * material.baseColor;
    
    // glTF stores roguhness in the G channel, metallic in the B channel and AO in the R channel
    float roughness = material.roughness * RMA.g;
    float metallic = material.metallic * RMA.b;
    float occlusion = RMA.r;

    
    SurfaceData surfData;
    surfData.N = normal;
    surfData.V = V;
    surfData.NdotV = saturate(dot(surfData.N, surfData.V));
    surfData.c_diff = lerp(diffuse.rgb, float3(0, 0, 0), metallic) * occlusion;
    surfData.c_spec = lerp(kDielectricSpecular, diffuse.rgb, metallic) * occlusion;
    
    float3 lRes = float3(0.0f, 0.0f, 0.0f);

   
    for (int i = 0; i < cNumLights; i++)
    {

        Light light = gLights[i];
        
        float shadowFactor = 1.0f;
        
        if (light.castShadows)
        {
            float4 shadowPosH = mul(float4(worldPos.xyz, 1.0f), light.shadowMatrix);
            shadowFactor = CalcShadowFactor(shadowPosH);
        }
        
        light.color *= shadowFactor;

        lRes += PBRDirectionalLight(light, surfData, roughness);
    }
    

    lRes += surfData.c_diff * 0.13f;

    
    return float4(lRes, diffuse.a);

}