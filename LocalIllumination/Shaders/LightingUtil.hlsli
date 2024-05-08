#define MAX_LIGHTS 16

struct Light
{
    float3 lColor;
    float lFallOffStart;
    float3 lDirection;
    float lFallOffEnd;
    float3 lPosition;
    float lSpotPower;
};

struct Material
{
    float4 diffuseColor;
    float4 specularColor;
    float4 ambientColor;
    float4 emissiveColor;
    
    float opacity;
    float shininess;
    float refractiveIndex;
    float bumpIntensity;
};

static const float3 ambientLightStrength = float3(0.2f, 0.2f, 0.3f);

struct LightResult
{
    float3 diffuse;
    float3 specular;
    float3 ambient;
};

// Light data utility structured used to pass light data to various lighting functions
struct LightData
{
    float3 L; // Light vector (negative of light direction)
    float3 NdotL; // Dot product of normal and light vector
    float3 LdotH; // Dot product of light vector and half vector
    float3 NdotH; // Dot product of normal and half vector
};

struct SurfaceData
{
    float3 normal;
    float3 toEye;
    float NdotV;
    float shininess;
    float refractiveIndex;
    float3 diffuseCol;
};

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

// Compute normal unit vector from two x and y components
float3 ComputeTwoChannelNormal(float2 normal)
{
    // Change normal mapping from [0, 1] to [-1, 1]
    float2 xy = 2.0f * normal - 1.0f;
    
    // Compute z from x and y
    float z = sqrt(1.0f - dot(xy, xy));

    return float3(xy.x, xy.y, z);
}

float3 SchlickFresnel3(float shininess, float3 normal, float3 lightVec)
{
    float3 R0 = (shininess-1) / (shininess + 1);
    R0 = R0 * R0;
    
    float f0 = 1.0f - max(dot(normal, lightVec), 0);
    
    return R0 + (1.0f - R0) * pow(f0, 5.0f);
}

float SchlickFresnel(float shininess, float3 normal, float3 lightVec)
{
    float R0 = (shininess - 1) / (shininess + 1);
    R0 = R0 * R0;
    
    float f0 = 1.0f - max(dot(normal, lightVec), 0);
    
    return R0 + (1.0f - R0) * pow(f0, 5.0f);
}

float3 Specular(float3 lightVec, SurfaceData surfData)
{    
    // Light vector is the opposite of the light direction
    float3 halfVec = normalize(lightVec + surfData.toEye);
    
    float roughness = (surfData.shininess + 8.0f) * pow(max(dot(halfVec, surfData.normal), 0.0f), surfData.shininess) / 8.0f;
    float3 fresnelFactor = SchlickFresnel3(surfData.shininess, surfData.normal, lightVec);
    
    float3 spec = fresnelFactor * roughness;
    
    spec = spec / (spec + 1.0f);
    
    return spec;
}

LightResult ComputeDirectionalLight(Light L, SurfaceData surfData)
{
    float3 lightVec = -L.lDirection;
    
    float3 NdotL = max(dot(surfData.normal, lightVec), 0.0f);
    
    LightResult lres;
    lres.diffuse = L.lColor * NdotL;
    lres.specular = Specular(lightVec, surfData);
    lres.ambient = ambientLightStrength;
    
    return lres;

}

//float3 Diffuse_Burley(Material mat, LightData lightData)
//{
//    float roughness = 1.0f - sqrt(1.0f / (1.0f + mat.shininess));
    
//    float fd90 = 0.5f * 2.0f * roughness * lightData.LdotH * lightData.LdotH;
    
//    return

//}

//float3 Specular_BRDF(Material mat, LightData lightData)
//{

//}

//float3 ComputeDirectionalLight(Material mat, Light dirLight, float3 surfaceNormal, float3 toEye)
//{
//    LightData lightData;
//    lightData.L = -dirLight.lDirection;
    
    
//    float3 halfVector = normalize(lightData.L + toEye);
    
//    lightData.NdotL = saturate(dot(surfaceNormal, lightData.L));
//    lightData.LdotH = saturate(dot(lightData.L, halfVector));
//    lightData.NdotH = saturate(dot(surfaceNormal, halfVector));
    
//    float3 diffuse = Diffuse_Burley(mat, lightData);
//    float3 specular = Specular_BRDF(mat, lightData);
    
//    return (diffuse + specular) * lightData.NdotL * dirLight.lColor;
//}