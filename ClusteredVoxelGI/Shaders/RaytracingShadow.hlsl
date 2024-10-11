#ifndef RAYTRACING_SHADOW_HLSL
#define RAYTRACING_SHADOW_HLSL

#define HLSL
#include "TechniquesCompat.h"
#include "RaytracingUtils.hlsli"
#include "VoxelUtils.hlsli"

ConstantBuffer<ConstantBufferVoxelCommons> cbVoxelCommons : register(b0);
ConstantBuffer<ConstantBufferRTShadows> cbRaytracingShadows : register(b1);

RaytracingAccelerationStructure Scene : register(t0, space0);

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space1);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space1);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space1);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space1);

StructuredBuffer<uint2> gVoxelFaceDataBuffer : register(t0, space2);

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

float4x3 GetFaceEdgeMidpoints(float3 voxelCenter, float3 faceDirection)
{
    // Define the half-size offsets.
    float3 right = float3(0.5f, 0.0f, 0.0f);
    float3 up = float3(0.0f, 0.5f, 0.0f);
    float3 forward = float3(0.0f, 0.0f, 0.5f);

    // Initialize a float3x4 to store the edge midpoints.
    float4x3 edgeMidpoints;

    // Determine midpoints based on face direction.
    if (faceDirection.x != 0.0f)
    {
        // Left or right face.
        edgeMidpoints[0] = faceDirection - up;
        edgeMidpoints[1] = faceDirection + up;
        edgeMidpoints[2] = faceDirection - forward;
        edgeMidpoints[3] = faceDirection + forward;
    }
    else if (faceDirection.y != 0.0f)
    {
        // Top or bottom face.
        edgeMidpoints[0] = faceDirection - right;
        edgeMidpoints[1] = faceDirection + right;
        edgeMidpoints[2] = faceDirection - forward;
        edgeMidpoints[3] = faceDirection + forward;
    }
    else if (faceDirection.z != 0.0f)
    {
        // Front or back face.
        edgeMidpoints[0] = faceDirection - right;
        edgeMidpoints[1] = faceDirection + right;
        edgeMidpoints[2] = faceDirection - up;
        edgeMidpoints[3] = faceDirection + up;
    }

    return edgeMidpoints;
}

[shader("raygeneration")]
void ShadowRaygen()
{
    float3 faceDirection[6] =
    {
        float3(0.0f, 0.0f, -0.5f),
        float3(0.0f, 0.0f, 0.5f),
        float3(-0.5f, 0.0f, 0.0f),
        float3(0.5f, 0.0f, 0.0f),
        float3(0.0f, -0.5f, 0.0f),
        float3(0.0f, 0.5f, 0.0f),
    };
    
    uint3 dispatchIdx = DispatchRaysIndex();
    uint3 dispatchDim = DispatchRaysDimensions();
    
    uint faceIdx = GetLinearCoord(dispatchIdx, dispatchDim);
    
    if (faceIdx >= cbRaytracingShadows.FaceCount)
        return;
    
    uint2 faceData = gVoxelFaceDataBuffer[faceIdx];
    
    float3 voxelCenter = float3(GetVoxelPosition(gVoxelHashedCompactBuffer[faceData.x], cbVoxelCommons.voxelTextureDimensions));
    
    
    // Define ray starting position and other parameters
    RayDesc ray;
    // A face is half a voxel distant from the center, so we divide the face direction by 2. We also multiply by 1.1 to add
    // a small offset to avoid self intersection
    ray.Direction = -cbRaytracingShadows.LightDirection;
    ray.TMin = 0.001; // Start the ray slightly away from the origin
    ray.TMax = 1.#INF; // Max distance for the ray
    Payload rayPayload = { 1 }; // Payload to store the index
    
    float3 points[5];
    float3 faceDir = faceDirection[faceData.y] * 1.05f;
    points[0] = voxelCenter + faceDir;
    {
        float4x3 edgeMidPoints = GetFaceEdgeMidpoints(voxelCenter, faceDir);
        points[1] = voxelCenter + edgeMidPoints[0];
        points[2] = voxelCenter + edgeMidPoints[1];
        points[3] = voxelCenter + edgeMidPoints[2];
        points[4] = voxelCenter + edgeMidPoints[3];
    }

    // Not much difference when tracing a ray from only the center of the face
    // Or also from the edges midpoints.
    for (uint i = 0; i < 1; i++)
    {
        ray.Origin = points[i];
        if (!IsVoxelPresent(faceData.x, gVoxelShadowBuffer))
        {
            TraceRay(Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, ~0, 0, 0, 0, ray, rayPayload);
            if (rayPayload.voxelIdx == 0)
            {
                SetVoxelPresence(faceData.x, gVoxelShadowBuffer);
                break;
            }
        }
        else
        {
            break;
        }
    }
}

[shader("miss")]
void ShadowMiss(inout Payload rayPayload)
{
    rayPayload.voxelIdx = 0;
}

[shader("intersection")]
void ShadowIntersection()
{
    Attributes attr = { 1 };
    ReportHit(1.0f, 0, attr); // Report the hit with hitT as tMin
}

#endif // RAYTRACING_SHADOW_HLSL
