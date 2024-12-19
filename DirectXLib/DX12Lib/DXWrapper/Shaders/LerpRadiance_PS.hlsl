#include "Common.hlsli"

struct ConstantBufferLerpRadiance
{
    uint CurrentPhase;
    uint FaceCount;
    uint2 screenDimension;
    
    float accumulatedTime;
    float maxTime;
    float pad0;
    float pad1;
};

cbuffer cbLerpRadiance : register(b0)
{
    ConstantBufferLerpRadiance lerpRadiance;
};

StructuredBuffer<uint2> gGaussianRadiance: register(t0);

RWStructuredBuffer<uint2> gRadianceLerpResult : register(u0);
RWStructuredBuffer<uint2> gLerpStartRadiance : register(u1);
RWStructuredBuffer<uint2> gLerpEndRadiance : register(u2);


float3 easeOutInterpolation(float3 colorA, float3 colorB, float t)
{
    // Clamp t to the range [0, 1] to avoid unexpected results
    t = saturate(t);

    // Apply the ease-out cubic formula: 1 - (1 - t)^4
    float easeT = 1.0 - pow(1.0 - t, 3.0);

    // Interpolate between the two colors using the eased t value
    return lerp(colorA, colorB, easeT);
}


float4 PS(VertexOutPosTex pIn) : SV_Target
{
    uint2 pixelCoord = uint2(pIn.Tex * lerpRadiance.screenDimension);
    uint idx = lerpRadiance.screenDimension.x * pixelCoord.y + pixelCoord.x;
    
    uint2 lerpStartRadiance = gLerpStartRadiance[idx];
    uint2 lerpEndRadiance = gLerpEndRadiance[idx];

    if (lerpRadiance.CurrentPhase == 1)
    {
        lerpStartRadiance = lerpEndRadiance;
        lerpEndRadiance = gGaussianRadiance[idx];
        
        gLerpStartRadiance[idx] = lerpStartRadiance;
        gLerpEndRadiance[idx] = lerpEndRadiance;
    }
    
    float3 lerpStartRadianceFloat = float3(0.0f, 0.0f, 0.0f);
    float3 lerpEndRadianceFloat = float3(0.0f, 0.0f, 0.0f);
    
    lerpStartRadianceFloat.xy = UnpackFloats16(lerpStartRadiance.x);
    lerpStartRadianceFloat.z = UnpackFloats16(lerpStartRadiance.y).x;
    
    lerpEndRadianceFloat.xy = UnpackFloats16(lerpEndRadiance.x);
    lerpEndRadianceFloat.z = UnpackFloats16(lerpEndRadiance.y).x;
    

    float t = saturate(lerpRadiance.accumulatedTime / lerpRadiance.maxTime);
    float3 lerpRadiance = easeOutInterpolation(lerpStartRadianceFloat, lerpEndRadianceFloat, t);
    
    uint2 lerpRadiancePacked = uint2(PackFloats16(lerpRadiance.xy), PackFloats16(float2(lerpRadiance.z, 0.0f)));
    
    gRadianceLerpResult[idx] = lerpRadiancePacked;
    
    return float4(0.0f, 0.0f, 0.0f, 0.0f);
}
