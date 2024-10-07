#ifndef RAYTRACING_SHADOW_HLSL
#define RAYTRACING_SHADOW_HLSL

#define HLSL
#include "TechniquesCompat.h"
#include "RaytracingUtils.hlsli"

SamplerComparisonState gShadowSampler : register(s0);

ConstantBuffer<ConstantBufferRTShadows> cbRaytracingShadows : register(b0);

cbuffer cbCameraBuffer : register(b1)
{
    float4x4 view;
    float4x4 invView;
    float4x4 projection;
    float4x4 invProjection;
    float4x4 viewProjection;
    float4x4 invViewProjection;
    
    float3 eyePosition;
    float nearPlane;
    
    float farPlane;
    float _pad0;
    float _pad1;
    float _pad2;
};

RaytracingAccelerationStructure Scene : register(t0, space0);

Texture2D<float> gShadowMap : register(t1, space0);

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space1);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space1);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space1);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space1);

StructuredBuffer<AABB> gVoxelAABBBuffer : register(t0, space2);
StructuredBuffer<AABBInfo> gClusterAABBInfoBuffer : register(t1, space2);
StructuredBuffer<uint> gAABBVoxelIndices : register(t2, space2);

StructuredBuffer<uint> gGeometryStartOffset : register(t0, space3);
StructuredBuffer<uint> gAABBStartOffset : register(t1, space3);

RWByteAddressBuffer gVoxelShadowBuffer : register(u0, space0);
RWByteAddressBuffer gTaaVoxelShadowBuffer : register(u1, space0);



struct Payload
{
    uint voxelIdx;
};

struct Attributes
{
    uint voxelIdx;
};

float Random(float2 seed)
{
    return frac(sin(dot(seed.xy, float2(12.9898, 78.233))) * 43758.5453);
}

static const float JITTER_AMOUNT = 0.01f;

inline void GenerateOrthoCameraRay(float2 ndc, out float3 origin, out float3 direction)
{
    float4 screenPos = float4(ndc * float2(240.0f, 240.0f), 0.0f, 1.0f);
    
    // Orthobounds are 120, 120, but the voxel scene is bigger than the raster scene by a factor of 2
    float3 worldPos = mul(screenPos, invView).xyz;
    
    // Jittering factor (scales the amount of jitter, adjust this value as needed)

    float jitterDir = Random(float2(ndc.y, ndc.x) + 2.0f) * 2.0f - 1;
    jitterDir *= 0.1f;
    
    origin = worldPos + eyePosition + cbRaytracingShadows.GridDimension / 2;
    
    direction = normalize(mul(float3(0.0f, 0.0f, 1.0f), (float3x3) invView));
}


void SetShadowVoxel(uint i)
{
    uint index = i / 32; // Calculate the 32-bit word index
    uint bitPosition = i % 32; // Calculate the bit position within the word
    
    // Use InterlockedOr to set the bit
    uint offset = index * 4; // Each 32-bit word is 4 bytes
    
    uint outValue = 0;
    uint taaOut = 0;
    
    gVoxelShadowBuffer.InterlockedOr(offset, (1u << bitPosition), outValue);

}

[shader("raygeneration")]
void ShadowRaygen()
{
    float2 rayOffset = float2(cbRaytracingShadows.ShadowTexDimensions.xy) / float2(DispatchRaysDimensions().xy);
    float2 rayStart = float2(DispatchRaysIndex().xy) * rayOffset;

    // Generate a 2D random offset
    float2 jitterOffset = float2(Random(rayStart), Random(rayStart + 1.0f));
    jitterOffset = jitterOffset * 2.0f - 1.0f;
    // Scale the jitter to make it a small offset
    jitterOffset = (jitterOffset) * JITTER_AMOUNT; // Center it around 0 and scale
        
    float2 ndc = rayStart / cbRaytracingShadows.ShadowTexDimensions; // normalized [0, 1]
    ndc = ndc * 2.0f - 1.0f; // normalized [-1, 1]
    ndc.y *= -1.0f; // flip y-axis
    
    
    float3 origin, direction;
    GenerateOrthoCameraRay(ndc, origin, direction);
        
        // Define ray starting position and other parameters
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.001; // Start the ray slightly away from the origin
    ray.TMax = 1.#INF; // Max distance for the ray
    Payload rayPayload = { -1 }; // Payload to store the index
        
    TraceRay(Scene, RAY_FLAG_NONE, ~0, 0, 0, 0, ray, rayPayload);
        
    uint idx = rayPayload.voxelIdx;
        
    if (idx == -1)
        return;
        
    SetShadowVoxel(idx);
       
    GenerateOrthoCameraRay(ndc, origin, direction);
    
    ray.Origin = origin;
    ray.Direction = direction + jitterOffset.x + jitterOffset.y;
    
    rayPayload.voxelIdx = -1;
    TraceRay(Scene, RAY_FLAG_NONE, ~0, 0, 0, 0, ray, rayPayload);
    
    idx = rayPayload.voxelIdx;
        
    if (idx == -1)
        return;
        
    SetShadowVoxel(idx);

}

[shader("closesthit")]
void ShadowClosestHit(inout Payload rayPayload, in Attributes attribs)
{
    uint idx = gAABBVoxelIndices[attribs.voxelIdx];
    
    rayPayload.voxelIdx = idx;
}

[shader("miss")]
void ShadowMiss(inout Payload rayPayload)
{
    rayPayload.voxelIdx = -1;
}

[shader("intersection")]
void ShadowIntersection()
{
    uint idx = gGeometryStartOffset[InstanceIndex()];
    idx += GeometryIndex();
    
    idx = gAABBStartOffset[idx] + PrimitiveIndex();
    
    AABB aabb = gVoxelAABBBuffer[idx];

    float3 aabbs[2] = { aabb.Min, aabb.Max };


    float tMin;
    float tMax;
    
    // Perform AABB intersection test
    
    if (RayAABBIntersectionTest(WorldRayOrigin(), WorldRayDirection(), aabbs, tMin, tMax))
    {

        Attributes attr;
        attr.voxelIdx = idx;
        
        ReportHit(tMin, 0, attr); // Report the hit with hitT as tMin
    }
}

#endif // RAYTRACING_SHADOW_HLSL
