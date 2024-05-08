#include "Common.hlsli"



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

VertexOut VS(VertexIn vIn)
{   
    VertexOut vOut;
    
    float4 posW = mul(float4(vIn.PosL, 1.0f), oWorld);
    
    vOut.PosW = posW.xyz;
    vOut.NormalW = mul(vIn.NormalL, (float3x3)oWorld);
    vOut.PosH = mul(posW, cViewProj);
    vOut.Tex = vIn.Tex;
    
    return vOut;
}

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
    float4 specular = gMaterial.specularColor * gSpecularTex.Sample(gSampler, pIn.Tex);
    float shininess = gMaterial.shininess * gShininessTex.Sample(gSampler, pIn.Tex).r;
    
    SurfaceData surfData = { normal, toEyeW, saturate(dot(normal, toEyeW)), shininess, gMaterial.refractiveIndex, diffuse.rgb };

    LightResult lres = ComputeDirectionalLight(cDirLight, surfData);

    float3 result = diffuse.rgb * lres.diffuse + specular.rgb * lres.specular + ambient.rgb * lres.ambient;
    
    return float4(result, diffuse.a);
}