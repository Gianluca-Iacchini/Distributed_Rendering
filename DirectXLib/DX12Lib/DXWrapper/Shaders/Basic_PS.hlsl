#include "Common.hlsli"

float4 PS(VertexOut pIn) : SV_TARGET
{
    float opacity = gOpacity.Sample(gSampler, pIn.Tex).r;
    
#ifdef ALPHA_TEST
    if (opacity < 0.1f)
    {
        discard;
    }
#endif 
    
    pIn.NormalW = normalize(pIn.NormalW);
    
    // Get normal from normal map. Compute Z from X and Y.
    float3 normalMapSample = ComputeTwoChannelNormal(gNormalMap.Sample(gSampler, pIn.Tex).xy);
    // Compute TBN matrix from position, normal and texture coordinates.
    float3x3 tbn = CalculateTBN(pIn.PosW, pIn.NormalW, pIn.Tex);
    // Transform normal from tangent space to world space
    float3 normal = normalize(mul(normalMapSample, tbn));
    
    float3 V = normalize(cEyePos - pIn.PosW);
    
    PhongMaterial material = GetPhongMaterial(gMaterials[oMaterialIndex]);
    
    float4 diffuse = material.diffuseColor * gDiffuseTex.Sample(gSampler, pIn.Tex);
    float4 emissive = material.emissiveColor * gEmissiveTex.Sample(gSampler, pIn.Tex);
    float4 ambient = material.ambientColor * gAmbientTex.Sample(gSampler, pIn.Tex);
    float4 specular = gSpecularTex.Sample(gSampler, pIn.Tex);
    float shininess = material.shininess * gShininessTex.Sample(gSampler, pIn.Tex).r;

    
    if (any(material.specularColor.rgb))
    {
        specular *= material.specularColor;
    }
    
    SurfaceData surfData;
    surfData.N = normal;
    surfData.V = V;
    surfData.NdotV = saturate(dot(surfData.N, surfData.V));
    surfData.c_diff = diffuse.rgb;
    surfData.c_spec = specular.rgb;
    
    float3 lightRes = ComputeDirectionalLight(gLights[0], surfData, shininess, material.refractiveIndex);    
    
    return float4(lightRes, diffuse.a);
}