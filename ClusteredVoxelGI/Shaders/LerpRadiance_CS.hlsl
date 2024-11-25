#include "VoxelUtils.hlsli"

ConstantBuffer<ConstantBufferVoxelCommons> cbCommons : register(b0);
ConstantBuffer<ConstantBufferLerpRadiance> cbLerpRadiance : register(b1);

RWStructuredBuffer<uint2> gCurrentGaussianRadiance : register(u0, space0);

RWStructuredBuffer<uint2> gFinalLerpRadiance : register(u0, space1);
RWStructuredBuffer<uint2> gLerpStartRadiance : register(u1, space1);
//RWStructuredBuffer<uint2> gOldGaussianRadiance : register(u2, space1);
RWStructuredBuffer<uint2> gLerpEndRadiance : register(u2, space1);


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
    
    //uint2 currentRadiance = gCurrentGaussianRadiance[DTid.x];
    //uint2 lerpStartRadiance = gLerpStartRadiance[DTid.x];

    //float t = saturate(cbLerpRadiance.accumulatedTime / cbLerpRadiance.maxTime);
    
    //if (cbLerpRadiance.CurrentPhase == 1)
    //{
    //    uint2 oldGaussianRadiance = gOldGaussianRadiance[DTid.x];
    //    gLerpStartRadiance[DTid.x] = oldGaussianRadiance;
    //    gOldGaussianRadiance[DTid.x] = currentRadiance;
    //    lerpStartRadiance = oldGaussianRadiance;
    //}
    
    //float3 currentRadianceFloat = float3(0.0f, 0.0f, 0.0f);
    //float3 lerpStartRadianceFloat = float3(0.0f, 0.0f, 0.0f);
    
    //currentRadianceFloat.xy = UnpackFloats16(currentRadiance.x);
    //currentRadianceFloat.z = UnpackFloats16(currentRadiance.y).x;
    
    //lerpStartRadianceFloat.xy = UnpackFloats16(lerpStartRadiance.x);
    //lerpStartRadianceFloat.z = UnpackFloats16(lerpStartRadiance.y).x;
    

    
    //float3 lerpRadiance = easeOutInterpolation(lerpStartRadianceFloat, currentRadianceFloat, t);
    
    //uint2 lerpRadiancePacked = uint2(PackFloats16(lerpRadiance.xy), PackFloats16(float2(lerpRadiance.z, 0.0f)));
    
    //gFinalLerpRadiance[DTid.x] = lerpRadiancePacked;
    
    
    uint2 lerpStartRadiance = gLerpStartRadiance[DTid.x];
    uint2 lerpEndRadiance = gLerpEndRadiance[DTid.x];

    if (cbLerpRadiance.CurrentPhase == 1)
    {
        lerpStartRadiance = lerpEndRadiance;
        lerpEndRadiance = gCurrentGaussianRadiance[DTid.x];
        
        gLerpStartRadiance[DTid.x] = lerpStartRadiance;
        gLerpEndRadiance[DTid.x] = lerpEndRadiance;
    }
    
    float3 lerpStartRadianceFloat = float3(0.0f, 0.0f, 0.0f);
    float3 lerpEndRadianceFloat = float3(0.0f, 0.0f, 0.0f);
    
    lerpStartRadianceFloat.xy = UnpackFloats16(lerpStartRadiance.x);
    lerpStartRadianceFloat.z = UnpackFloats16(lerpStartRadiance.y).x;
    
    lerpEndRadianceFloat.xy = UnpackFloats16(lerpEndRadiance.x);
    lerpEndRadianceFloat.z = UnpackFloats16(lerpEndRadiance.y).x;
    

    float t = saturate(cbLerpRadiance.accumulatedTime / cbLerpRadiance.maxTime);
    float3 lerpRadiance = easeOutInterpolation(lerpStartRadianceFloat, lerpEndRadianceFloat, t);
    
    uint2 lerpRadiancePacked = uint2(PackFloats16(lerpRadiance.xy), PackFloats16(float2(lerpRadiance.z, 0.0f)));
    
    gFinalLerpRadiance[DTid.x] = lerpRadiancePacked;
}