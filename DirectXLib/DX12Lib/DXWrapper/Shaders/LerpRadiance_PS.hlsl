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


RWStructuredBuffer<uint2> gLerpStartRadiance : register(u0);
RWStructuredBuffer<uint2> gLerpEndRadiance : register(u1);
RWStructuredBuffer<uint2> gRadianceLerpResult : register(u2);

float3 easeOutInterpolation(float3 colorA, float3 colorB, float t)
{
    // Clamp t to the range [0, 1] to avoid unexpected results
    t = saturate(t);

    // Apply the ease-out cubic formula: 1 - (1 - t)^4
    float easeT = 1.0 - pow(1.0 - t, 3.0);

    // Interpolate between the two colors using the eased t value
    return lerp(colorA, colorB, easeT);
}


void UpdateRadiance(uint idx)
{
    uint2 lerpStartRadiance = gLerpStartRadiance[idx];
    uint2 lerpEndRadiance = gLerpEndRadiance[idx];

    if (lerpRadiance.CurrentPhase == 1)
    {
        lerpStartRadiance = gRadianceLerpResult[idx];
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
}

// We don't write to any render target during this pass.
// This proved to be faster than using a compute shader.
void PS(VertexOutPosTex pIn)
{
    uint totPixels = lerpRadiance.screenDimension.x * lerpRadiance.screenDimension.y;
    
    uint facesPerPixel = uint(ceil(float(lerpRadiance.FaceCount) / float(totPixels)));
    
    uint2 pixelCoord = uint2(pIn.Tex * lerpRadiance.screenDimension);
    uint startIdx = lerpRadiance.screenDimension.x * pixelCoord.y + pixelCoord.x;
    
    startIdx *= facesPerPixel;
    uint endIdx = min(startIdx + facesPerPixel, lerpRadiance.FaceCount);
    
    for (uint idx = startIdx; idx < endIdx; idx++)
    {
        UpdateRadiance(idx);
    }
}
