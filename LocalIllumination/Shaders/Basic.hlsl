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
    
    float4 posW = mul(float4(vIn.PosL, 1.0f), world);
    
    vOut.PosW = posW.xyz;
    vOut.NormalW = mul(vIn.NormalL, (float3x3)world);
    vOut.PosH = mul(posW, viewProj);
    vOut.Tex = vIn.Tex;
    
    return vOut;
}

float4 PS(VertexOut pIn) : SV_TARGET
{
    float4 diffuseColor = diffuse;
    diffuseColor *= diffuseTex.Sample(gSampler, pIn.Tex);
    
    pIn.NormalW = normalize(pIn.NormalW);
    
    // Get normal from normal map. Compute Z from X and Y.
    float3 normalMapSample = ComputeTwoChannelNormal(normalMap.Sample(gSampler, pIn.Tex).xy);
    // Compute TBN matrix from position, normal and texture coordinates.
    float3x3 tbn = CalculateTBN(pIn.PosW, pIn.NormalW, pIn.Tex);
    // Transform normal from tangent space to world space
    float3 normal = normalize(mul(normalMapSample, tbn));
    
    //return float4(normalize(pIn.NormalW), 1.0f);
    //return diffuseTex.Sample(gSampler, pIn.Tex);
    
    return float4(normal.xyz, 1);

}