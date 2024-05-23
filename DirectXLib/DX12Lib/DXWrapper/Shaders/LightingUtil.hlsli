#define MAX_LIGHTS 16

struct Light
{
    float3 color;
    float fallOffStart;
    float3 direction;
    float fallOffEnd;
    float3 position;
    float spotPower;
};

struct GenericMaterial
{
    float4 float4_0;
    float4 float4_1;
    float4 float4_2;
    float4 float4_3;
    
    float float_0;
    float float_1;
    float float_2;
    float float_3;
};


struct PBRMaterial
{
    float4 baseColor;
    float4 emissiveColor;
    
    float metallic;
    float roughness;
    float normalScale;
};

struct PhongMaterial
{
    float4 diffuseColor;
    float4 emissiveColor;
    float4 specularColor;
    float4 ambientColor;

    float normalScale;
    float opacity;
    float shininess;
    float refractiveIndex;
};

PBRMaterial GetPBRMaterial(GenericMaterial material)
{
    PBRMaterial pbrMaterial;
    
    pbrMaterial.baseColor = material.float4_0;
    pbrMaterial.emissiveColor = material.float4_1;
    
    pbrMaterial.normalScale = material.float_0;
    pbrMaterial.metallic = material.float_1;
    pbrMaterial.roughness = material.float_2;

    return pbrMaterial;
}

PhongMaterial GetPhongMaterial(GenericMaterial material)
{
    PhongMaterial phongMaterial;
    
    phongMaterial.diffuseColor = material.float4_0;
    phongMaterial.emissiveColor = material.float4_1;
    phongMaterial.specularColor = material.float4_2;
    phongMaterial.ambientColor = material.float4_3;
    
    phongMaterial.normalScale = material.float_0;
    phongMaterial.opacity = material.float_1;
    phongMaterial.shininess = material.float_2;
    phongMaterial.refractiveIndex = material.float_3;
    
    return phongMaterial;
}

static const float3 ambientLightStrength = float3(0.13f, 0.13f, 0.13f);

struct LightResult
{
    float3 diffuse;
    float3 specular;
    float3 ambient;
};

struct SurfaceData
{
    float3 N;
    float3 V;
    float NdotV;
    float3 c_diff;
    float3 c_spec;
};

static const float PI = 3.14159265;
static const float3 kDielectricSpecular = float3(0.04, 0.04, 0.04);
static const float EPSILON = 1e-6f;

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


//float3 SchlickFresnel3(float refractiveIndex, float3 normal, float3 lightVec)
//{
//    float3 R0 = (refractiveIndex-1) / (refractiveIndex + 1);
//    R0 = R0 * R0;
    
//    float f0 = 1.0f - saturate(dot(normal, lightVec));
    

//    return R0 + (1.0f - R0) * pow(f0, 5.0f);
//}

float Spec(float3 lightVec, SurfaceData surfData, float shininess)
{
    float3 R = normalize(reflect(-lightVec, surfData.N));
    float RdotV = max(0, dot(R, surfData.V));
    
    return pow(RdotV, shininess);
}

//float3 Specular(float3 lightVec, SurfaceData surfData, float shininess, float IoR)
//{    
//    // Light vector is the opposite of the light direction
//    float3 halfVec = normalize(lightVec + surfData.N);
    
//    float roughness = (shininess + 8.0f) * pow(max(dot(halfVec, surfData.N), 0.0f), shininess) / 8.0f;
//    float3 fresnelFactor = SchlickFresnel3(IoR, surfData.N, lightVec);
    
//    float3 spec = fresnelFactor * roughness;
    
//    spec = spec / (spec + 1.0f);
    
//    return spec;
//}

float3 ComputeDirectionalLight(Light light, SurfaceData surfData, float shininess, float IoR)
{
    float3 L = -light.direction;
    
    float3 NdotL = max(dot(surfData.N, L), 0.0f);
    

    float3 diffuse = light.color * NdotL;
    float3 specular = light.color * Spec(L, surfData, shininess);

   
    diffuse = saturate(diffuse);
    specular = saturate(specular);
    
    
    float3 ambient = surfData.c_diff * ambientLightStrength;
    ambient = saturate(ambient);
        
    return (diffuse * surfData.c_diff + specular * surfData.c_spec + ambient);
}

float3 FresnelShlick(float3 F0, float3 F90, float cosine)
{
    return F0 + (F90 - F0) * pow(1.f - cosine, 5.f);

}

float3 Diffuse_Burley(SurfaceData surfData, float roughness, float LdotH, float NdotL)
{
    float fd90 = 0.5 + 2.0 * roughness * LdotH * LdotH;
    
    return FresnelShlick(1, fd90, NdotL).x * FresnelShlick(1, fd90, surfData.NdotV).x;
}

float Specular_D_GGX(SurfaceData surfData, float NdotH, float alphaSqr)
{

    float lower = (NdotH * NdotH * (alphaSqr - 1)) + 1;
    
    return alphaSqr / max(EPSILON, PI * lower * lower);

}

float G_Shlick_Smith_Hable(SurfaceData surfData, float LdotH, float alphaSqr)
{
    return rcp(lerp(LdotH * LdotH, 1, alphaSqr * 0.25f));
}

float3 Specular_BRDF(SurfaceData surfData, float LdotH, float NdotH, float alpha)
{
    float alphaSqr = alpha * alpha;
    
    float ND = Specular_D_GGX(surfData, NdotH, alphaSqr);
    
    float GV = G_Shlick_Smith_Hable(surfData, LdotH, alphaSqr);
    
    float3 F = FresnelShlick(surfData.c_spec, 1.0, LdotH);
    
    return ND * F * GV;

}



float3 PBRDirectionalLight(Light Light, SurfaceData surfData, float roughness)
{    
    float3 L = normalize(-Light.direction);
    
    
    float3 H = normalize(L + surfData.V);
    
    float NdotL = saturate(dot(surfData.N, L));
    float LdotH = saturate(dot(L, H));
    float NdotH = saturate(dot(surfData.N, H));
    
    float alpha = roughness * roughness;
    
    float3 diffuse = Diffuse_Burley(surfData, roughness, LdotH, NdotL);
    float3 specular = Specular_BRDF(surfData, LdotH, NdotH, alpha);
    
   return NdotL * Light.color * ((surfData.c_diff * diffuse) + specular);
}