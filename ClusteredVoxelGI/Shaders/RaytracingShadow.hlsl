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

StructuredBuffer<ClusterData> gClusterDataBuffer : register(t0, space2);
StructuredBuffer<uint> gNextVoxelLinkedList : register(t1, space2);
StructuredBuffer<uint> gVoxelAssignmentMap : register(t2, space2);
StructuredBuffer<float3> gVoxelColorBuffer : register(t3, space2);


RWByteAddressBuffer gVoxelLitBuffer : register(u0, space0);
RWStructuredBuffer<uint4> gClusterLitBuffer : register(u1, space0);



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
    float3 right = float3(1.0f, 0.0f, 0.0f);
    float3 up = float3(0.0f, 1.0f, 0.0f);
    float3 forward = float3(0.0f, 0.0f, 1.0f);

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

    uint3 dispatchIdx = DispatchRaysIndex();
    uint3 dispatchDim = DispatchRaysDimensions();

    float3 faceDirectionPos[3] =
    {
        float3(0.0f, 0.0f, 0.5f),
        float3(0.5f, 0.0f, 0.0f),
        float3(0.0f, 0.5f, 0.0f),
    };
    
    uint linearIdx = GetLinearCoord(dispatchIdx, dispatchDim);
    
    if (linearIdx >= cbRaytracingShadows.FaceCount)
        return;
    
    // At most, only 3 faces can be lit by the light source
    uint voxelIdx = (uint) floor(linearIdx / 3.0f);
    uint faceIdx = linearIdx % 3;
    
    float3 faceDir = faceDirectionPos[faceIdx];
    faceDir *= sign(dot(-cbRaytracingShadows.LightDirection, faceDir));
    
    float3 voxelCenter = float3(GetVoxelPosition(gVoxelHashedCompactBuffer[voxelIdx], cbVoxelCommons.voxelTextureDimensions));
    
    
    // Define ray starting position and other parameters
    RayDesc ray;
    // A face is half a voxel distant from the center, so we divide the face direction by 2. We also multiply by 1.1 to add
    // a small offset to avoid self intersection
    ray.Direction = -cbRaytracingShadows.LightDirection;
    // Start the ray slightly away from the origin 
    // TODO: check if this value is okay or can be improved
    ray.TMin = 0.5f; 
    // Chebyshev Distance (1.74 is an approximation of sqrt(3));
    // This is an approximation of the scene diagonal, which is the maximum distance a ray can travel for a hit.
    ray.TMax = max(max(cbVoxelCommons.voxelTextureDimensions.x, cbVoxelCommons.voxelTextureDimensions.y), cbVoxelCommons.voxelTextureDimensions.z);
    ray.TMax *= 1.74f; 
    Payload rayPayload = { 1 }; // Payload to store the index
    
    float3 points[5];

    points[0] = voxelCenter + faceDir;
    {
        float4x3 edgeMidPoints = GetFaceEdgeMidpoints(voxelCenter, faceDir);
        points[1] = voxelCenter + edgeMidPoints[0];
        points[2] = voxelCenter + edgeMidPoints[1];
        points[3] = voxelCenter + edgeMidPoints[2];
        points[4] = voxelCenter + edgeMidPoints[3];
    }

    uint nSampleVisible = 0;
    
    // Not much difference when tracing a ray from only the center of the face
    // Or also from the edges midpoints.
    for (uint i = 0; i < 5; i++)
    {

        
        ray.Origin = points[i];

        TraceRay(Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, ~0, 0, 0, 0, ray, rayPayload);
        if (rayPayload.voxelIdx == 0)
        {
            nSampleVisible += 1;
        }
        
        if (nSampleVisible > 4)
            break;
        
        rayPayload.voxelIdx = 1;
    }
    
    if (nSampleVisible > 4)
    {
        bool wasAlreadyLit = SetVoxelPresence(voxelIdx, gVoxelLitBuffer);
        
        if (!wasAlreadyLit)
        {
            uint clusterIdx = gVoxelAssignmentMap[voxelIdx];
            
            if (clusterIdx != UINT_MAX)
            {
                ClusterData clusterData = gClusterDataBuffer[clusterIdx];
        
                float formFactor = differentialAreaFormFactor(clusterData.Normal, cbRaytracingShadows.LightDirection);

                float3 voxelRadiance = formFactor * clusterData.Color * 150.0f;
                uint3 irradianceUint = uint3(voxelRadiance * IRRADIANCE_FIELD_MULTIPLIER);
                
                InterlockedAdd(gClusterLitBuffer[clusterIdx].x, irradianceUint.x);
                InterlockedAdd(gClusterLitBuffer[clusterIdx].y, irradianceUint.y);
                InterlockedAdd(gClusterLitBuffer[clusterIdx].z, irradianceUint.z);
                InterlockedAdd(gClusterLitBuffer[clusterIdx].w, 1);
            }
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
