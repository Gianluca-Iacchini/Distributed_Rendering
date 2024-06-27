#define PBR 1
#include "Common.hlsli"

struct PSOut
{
    float4 GBufferA : SV_Target0;
    float2 GBufferB : SV_Target1;
    float4 GBufferC : SV_Target2;
    float4 GBufferD : SV_Target3;
};

PSOut PS(VertexOut pIn)
{
    PSOut psOut;
    
    pIn.NormalW = normalize(pIn.NormalW);
    
    // Get normal from normal map. Compute Z from X and Y.
    float3 normalMapSample = ComputeTwoChannelNormal(gNormalMap.Sample(gSampler, pIn.Tex).xy);
    // Compute TBN matrix from position, normal and texture coordinates.
    float3x3 tbn = CalculateTBN(pIn.PosW, pIn.NormalW, pIn.Tex);
    // Transform normal from tangent space to world space
    float3 normal = normalize(mul(normalMapSample, tbn));
    
    psOut.GBufferA = float4(pIn.PosW, oMaterialIndex);
    psOut.GBufferB = PackNormal(normal);
    //psOut.GBufferB = float4(normal, 1.0f);
    psOut.GBufferC = gDiffuseTex.Sample(gSampler, pIn.Tex);
    
    // w coordinated is used to keep track of geometry. Pixels where there is no geometry have a w value of 1.0f
    // due to the fact that the clear color used for this GBuffer is red (1.0f, 0.0f, 0.0f, 1.0f)
    psOut.GBufferD = float4(gMetallicRoughness.Sample(gSampler, pIn.Tex).rgb, 0.0f);
    
    return psOut;
}