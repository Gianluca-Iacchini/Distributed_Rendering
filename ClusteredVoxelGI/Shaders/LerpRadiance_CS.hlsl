#include "VoxelUtils.hlsli"

ConstantBuffer<ConstantBufferVoxelCommons> cbCommons : register(b0);
ConstantBuffer<ConstantBufferLerpRadiance> cbLerpRadiance : register(b1);

RWStructuredBuffer<uint2> gGaussianReadBuffer : register(u0, space0);

RWStructuredBuffer<uint2> gOldRadiance : register(u0, space1);
RWStructuredBuffer<uint2> gNewRadiance : register(u1, space1);

float3 easeOutInterpolation(float3 colorA, float3 colorB, float t)
{
    // Clamp t to the range [0, 1] to avoid unexpected results
    t = saturate(t);

    // Apply the ease-out cubic formula: 1 - (1 - t)^4
    float easeT = 1.0 - pow(1.0 - t, 3.0);

    // Interpolate between the two colors using the eased t value
    return lerp(colorA, colorB, easeT);
}

[numthreads(128, 1, 1)]
void CS( uint3 DTid : SV_DispatchThreadID )
{
    if (DTid.x >= cbLerpRadiance.FaceCount)
        return;
    
    if (cbLerpRadiance.accumulatedTime > cbLerpRadiance.maxTime)
        return;
    
    uint2 oldRadiancePacked = gOldRadiance[DTid.x];
    uint2 newRadiancePacked = gNewRadiance[DTid.x];
    
    if (cbLerpRadiance.CurrentPhase == 1)
    {
        oldRadiancePacked = newRadiancePacked;
        newRadiancePacked = gGaussianReadBuffer[DTid.x];
        
        gOldRadiance[DTid.x] = oldRadiancePacked;
        gNewRadiance[DTid.x] = newRadiancePacked;
    }
        
    float3 oldRadiance = float3(0.0f, 0.0f, 0.0f);
    float3 newRadiance = float3(0.0f, 0.0f, 0.0f);
    
    oldRadiance.xy = UnpackFloats16(oldRadiancePacked.x);
    oldRadiance.z = UnpackFloats16(oldRadiancePacked.y).x;
    
    newRadiance.xy = UnpackFloats16(newRadiancePacked.x);
    newRadiance.z = UnpackFloats16(newRadiancePacked.y).x;
        
    float lerpFactor = cbLerpRadiance.accumulatedTime / cbLerpRadiance.maxTime;
    
    float3 lerpRadiance = lerp(oldRadiance, newRadiance, lerpFactor);
    
    uint2 packedRadiance = uint2(0, 0);
    packedRadiance.x = PackFloats16(lerpRadiance.xy);
    packedRadiance.y = PackFloats16(float2(lerpRadiance.z, 0.0f));
    
    gGaussianReadBuffer[DTid.x] = packedRadiance;
    
}