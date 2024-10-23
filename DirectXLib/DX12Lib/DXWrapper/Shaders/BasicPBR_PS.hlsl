#include "Common.hlsli"

cbuffer cbCommons : register(b0)
{
    Commons commons;
};

cbuffer cbCamera : register(b1)
{
    Camera camera;
}


cbuffer cbVoxelTransform : register(b2)
{
    float4x4 voxelToWorld;
    float4x4 worldToVoxel;
}

Texture2D gShadowMap : register(t0);
Texture2D gVoxelSrv : register(t1);

Texture2D gBufferaWorld : register(t2);
Texture2D gBufferNormal : register(t3);
Texture2D gBufferDiffuse : register(t4);
Texture2D gBufferMetallicRoughnessAO : register(t5);

StructuredBuffer<Light> gLights : register(t0, space1);
StructuredBuffer<GenericMaterial> gMaterials : register(t1, space1);

StructuredBuffer<uint2> gPackedRadiance : register(t0, space2);

float4 PS(VertexOutPosTex pIn) : SV_Target
{
    
    float4 diffuse = gBufferDiffuse.Sample(gSampler, pIn.Tex);
    float4 worldPos = gBufferaWorld.Sample(gSampler, pIn.Tex);
    
    float2 normalPacked = gBufferNormal.Sample(gSampler, pIn.Tex).rg;
    float3 normal = UnpackNormal(normalPacked);
    float4 RMA = gBufferMetallicRoughnessAO.Sample(gSampler, pIn.Tex);
        
    // w coordinate is used to keep track of geometry. If there is no geometry, the w value is 1.0f so we can discard the pixel
    if (RMA.w >= 1.0f)
        discard;
     
    
    PBRMaterial material = GetPBRMaterial(gMaterials[worldPos.w]);
    
    float3 V = normalize(camera.EyePos - worldPos.xyz);
    

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

   
    for (int i = 0; i < commons.NumLights; i++)
    {

        Light light = gLights[i];
        
        float shadowFactor = 1.0f;
        
        if (light.castShadows)
        {
            float4 shadowPosH = mul(float4(worldPos.xyz, 1.0f), light.shadowMatrix);
            shadowFactor = CalcShadowFactor(gShadowMap, shadowPosH);
        }
        
        light.color *= shadowFactor;

        lRes += PBRDirectionalLight(light, surfData, roughness);
    }
    
    if (commons.UseRTGI == 0)
    {
        lRes += surfData.c_diff * 0.13f;
    }
    else
    {

    }
    
    return float4(lRes, diffuse.a);

}