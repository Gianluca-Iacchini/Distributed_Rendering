#include "Common.hlsli"

float4 PS(VertexOut pIn) : SV_TARGET
{

    
    pIn.NormalW = normalize(pIn.NormalW);
    
    // Get normal from normal map. Compute Z from X and Y.
    float3 normalMapSample = ComputeTwoChannelNormal(gNormalMap.Sample(gSampler, pIn.Tex).xy);
    // Compute TBN matrix from position, normal and texture coordinates.
    float3x3 tbn = CalculateTBN(pIn.PosW, pIn.NormalW, pIn.Tex);
    // Transform normal from tangent space to world space
    float3 normal = normalize(mul(normalMapSample, tbn));
    
    float3 toEyeW = normalize(cEyePos - pIn.PosW);
    
    float4 diffuse = gMaterial.diffuseColor * gDiffuseTex.Sample(gSampler, pIn.Tex);
    float4 ambient = gMaterial.ambientColor * gAmbientTex.Sample(gSampler, pIn.Tex);
    float4 specular = gSpecularTex.Sample(gSampler, pIn.Tex);
    float shininess = gMaterial.shininess * gShininessTex.Sample(gSampler, pIn.Tex).r;
    
    if (any(gMaterial.specularColor.rgb))
    {
        specular *= gMaterial.specularColor;
    }
    
    SurfaceData surfData = { normal, toEyeW, saturate(dot(normal, toEyeW)), shininess, gMaterial.refractiveIndex, diffuse.rgb };

    LightResult lres = ComputeDirectionalLight(cDirLight, surfData);

    float3 result = diffuse.rgb * lres.diffuse + specular.rgb * lres.specular + ambient.rgb * lres.ambient;
    
        
    return float4(result, diffuse.a);
}