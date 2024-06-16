#define PBR 1
#include "Common.hlsli"

float4 PS(VertexOut pIn) : SV_TARGET
{

    float4 diffuse = gDiffuseTex.Sample(gSampler, pIn.Tex);

#ifdef ALPHA_TEST
    if (diffuse.a < 0.1f)
    {
        discard;
    }
#endif
   
    
    PBRMaterial material = GetPBRMaterial(gMaterials[oMaterialIndex]);
    
    pIn.NormalW = normalize(pIn.NormalW);
    
    // Get normal from normal map. Compute Z from X and Y.
    float3 normalMapSample = ComputeTwoChannelNormal(gNormalMap.Sample(gSampler, pIn.Tex).xy);
    // Compute TBN matrix from position, normal and texture coordinates.
    float3x3 tbn = CalculateTBN(pIn.PosW, pIn.NormalW, pIn.Tex);
    // Transform normal from tangent space to world space
    float3 normal = normalize(mul(normalMapSample, tbn));
    
    float3 V = normalize(cEyePos - pIn.PosW);
    
    float4 emissive = material.emissiveColor * gEmissiveTex.Sample(gSampler, pIn.Tex);
    diffuse = diffuse * material.baseColor;
    
    // glTF stores roguhness in the G channel, metallic in the B channel and AO in the R channel
    float3 RMA = gMetallicRoughness.Sample(gSampler, pIn.Tex).rgb;
    
    float roughness = material.roughness * RMA.g;
    float metallic = material.metallic * RMA.b;
    float occlusion = RMA.r;

    
    SurfaceData surfData;
    surfData.N = normal;
    surfData.V = V;
    surfData.NdotV = saturate(dot(surfData.N, surfData.V));
    surfData.c_diff = lerp(diffuse.rgb, float3(0, 0, 0), metallic) * occlusion;
    surfData.c_spec = lerp(kDielectricSpecular, diffuse.rgb, metallic) * occlusion;
    
    float3 lRes = emissive.rgb;    
    Light light = gLights[0];
    
    if (cNumLights > 0)
    {

        
        float shadow = 1.0f;
        
        if (cUseShadows)
        {
            shadow = CalcShadowFactor(pIn.ShadowPosH);
            //float shadow = gShadowMap.SampleCmpLevelZero(gShadowSampler, pIn.ShadowPosH.xy, pIn.ShadowPosH.z).r;
            light.color *= shadow;
        }

        lRes += PBRDirectionalLight(light, surfData, roughness);
    }

    for (int i = 1; i < cNumLights; i++)
    {
        light = gLights[i];
        lRes += PBRDirectionalLight(light, surfData, roughness);
    }
    

    lRes += surfData.c_diff * 0.13f;

    
    return float4(lRes, diffuse.a);

}