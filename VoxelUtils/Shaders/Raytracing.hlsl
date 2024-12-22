//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#ifndef RAYTRACING_HLSL
#define RAYTRACING_HLSL

#include "../../VoxelUtils/Shaders/VoxelUtils.hlsli"
#include "RaytracingUtils.hlsli"

struct Vertex
{
    float3 position;
    float3 normal;
};


ConstantBuffer<RTSceneVisibility> g_sceneCB : register(b0);


RaytracingAccelerationStructure Scene : register(t0, space0);

ByteAddressBuffer gVoxelOccupiedBuffer : register(t0, space1);

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space2);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space2);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space2);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space2);


StructuredBuffer<uint> gNextVoxelInClusterBuffer : register(t1, space3);
StructuredBuffer<uint> gVoxelAssignmentBuffer : register(t2, space3);

StructuredBuffer<AABB> gVoxelAABBBuffer : register(t0, space4);
StructuredBuffer<ClusterAABBInfo> gClusterAABBInfoBuffer : register(t1, space4);
StructuredBuffer<uint> gAABBVoxelIndices : register(t2, space4);

RWStructuredBuffer<uint2> gFaceClusterVisibility : register(u0);
// Stores all the visible clusters for all the faces. Clusters visible from the same faced are stored in sequence.
RWStructuredBuffer<uint> gVisibleClustersBuffer : register(u1);

// Offset buffers used to map the raytracing primitive index to the voxel index
RWStructuredBuffer<uint> gGeometryStartOffset : register(u2);
RWStructuredBuffer<uint> gAABBStartOffset : register(u3);
// For face at index i store in the x coordinate the start index in gVisibleClusterBuffer and in the y coordinate the number of visible
// Clusters
RWStructuredBuffer<uint> gClusterCount : register(u4);


struct AABBAttributes
{
    uint aabbIndex;
};

struct RayPayload
{
    uint primitiveIndex;
};

float3 LinearIndexToColor(uint index)
{

    // Hash the index to produce a pseudo-random float3 color
    uint hash = index;

    // Example hash function (based on bitwise operations)
    hash = (hash ^ 61) ^ (hash >> 16);
    hash = hash + (hash << 3);
    hash = hash ^ (hash >> 4);
    hash = hash * 0x27d4eb2d;
    hash = hash ^ (hash >> 15);

    // Convert the hash to a float3 color in the range [0, 1]
    float r = (float) ((hash >> 16) & 0xFF) / 255.0;
    float g = (float) ((hash >> 8) & 0xFF) / 255.0;
    float b = (float) (hash & 0xFF) / 255.0;

    if (r < 0.1 && g < 0.1 && b < 0.1)
    {
        r = 0.39f;
        g = 0.39f;
        b = 0.39f;
    }
    
    return float3(r, g, b);
}

static const float FLT_MAX = 1.#INF;

// Retrieve hit world position.
float3 HitWorldPosition()
{
    return WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
}

// Retrieve attribute at a hit position interpolated from vertex attributes using the hit's barycentrics.
float3 HitAttribute(float3 vertexAttribute[3], BuiltInTriangleIntersectionAttributes attr)
{
    return vertexAttribute[0] +
        attr.barycentrics.x * (vertexAttribute[1] - vertexAttribute[0]) +
        attr.barycentrics.y * (vertexAttribute[2] - vertexAttribute[0]);
}


float rand2(float2 p)
{
    return frac(cos(dot(p, float2(23.14069263277926f, 2.665144142690225f))) * 12345.6789f);
}






float3x3 ComputeOrthoBasis(float3 direction)
{
    float3 Z = normalize(direction); // Z-axis is the given direction
    float3 up = abs(Z.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0); // Choose up vector
    float3 X = normalize(cross(up, Z)); // X-axis is perpendicular to Z
    float3 Y = cross(Z, X); // Y-axis is perpendicular to both X and Z
    return float3x3(X, Y, Z); // Return the local coordinate system
}

float3 UniformHemisphereSample(float u1, float u2)
{
    float theta = acos(u1); // Theta is the polar angle
    float phi = 2.0 * PI * u2; // Phi is the azimuth angle
    float x = sin(theta) * cos(phi); // Convert spherical to Cartesian coordinates
    float y = sin(theta) * sin(phi);
    float z = cos(theta);
    return float3(x, y, z); // Hemisphere direction
}


static uint clusterFound[800];

/** Uses the shared variable array g_sharedArrayIndexBitWise to flag already verified clusters
* and avoid counting two or more times irradiance to be added to a voxel from the same cluster
* @param indexHashed [in] index of the cluster to tag and verify if already occupied
* @return true if already tagged, false otherwise */
bool alreadyOccupiedVoxel(uint indexHashed)
{
    float integerPart;
    float indexDecimalFloat = float(indexHashed) / 32.0;
    float fractional = modf(indexDecimalFloat, integerPart);
    uint index = uint(integerPart);
    uint bit = uint(fractional * 32.0);
    uint value = (1 << bit);
    
    // I don't need a lock here because only this thread has access to clusterFound.
    uint previousValue = clusterFound[index];
    clusterFound[index] = previousValue | value;
	
	// Verify if the bit was previously flagged and decide for this thread
    bool result = ((previousValue & (1 << bit)) > 0);

    return result;
}

[shader("raygeneration")]
void MyRaygenShader()
{
    float3 faceDirection[6] =
    {
        float3(0.0f, 0.0f, -1.0f),
        float3(0.0f, 0.0f, 1.0f),
        float3(-1.0f, 0.0f, 0.0f),
        float3(1.0f, 0.0f, 0.0f),
        float3(0.0f, -1.0f, 0.0f),
        float3(0.0f, 1.0f, 0.0f),
    };
        
    uint3 dispatchId = DispatchRaysIndex();
    uint3 dispatchDim = DispatchRaysDimensions();
    
    uint linearIndex = dispatchId.x + dispatchId.y * dispatchDim.x + dispatchId.z * dispatchDim.x * dispatchDim.y;
    
    if (linearIndex >= g_sceneCB.FaceCount)
        return;
    
    uint voxelIndex = (uint) floor(linearIndex / 6.0f);
    uint faceIdx = linearIndex % 6;
    
    float3 voxelCoord = float3(GetVoxelPosition(gVoxelHashedCompactBuffer[voxelIndex], g_sceneCB.GridDimension));
    
    float3 neighbourCoord = voxelCoord + faceDirection[faceIdx];
    
    
    
    bool isOutOfBounds = any(neighbourCoord < 0.0f) || any(neighbourCoord >= g_sceneCB.GridDimension);
    bool isOccupied = IsVoxelPresent(uint3(neighbourCoord), g_sceneCB.GridDimension, gVoxelOccupiedBuffer);
    
    uint visibleClusterCount = 0;
    
    if (!isOutOfBounds && !isOccupied)
    {
        // Compute the orthonormal basis
        float3x3 basis = ComputeOrthoBasis(faceDirection[faceIdx]);
    
        // Define ray starting position and other parameters
        RayDesc ray;
        ray.Origin = voxelCoord + faceDirection[faceIdx] / 2.0f; /* define ray origin (e.g., surface point) */;
        ray.TMin = 0.001; // Start the ray slightly away from the origin
        ray.TMax = FLT_MAX; // Max distance for the ray
        RayPayload payload = { -1 }; // Payload to store the color
    

    
        // Trace the ray
        for (uint i = 0; i < 128; i++)
        {
            float div = i / 128.0f;
            float u1 = rand2(float2(g_sceneCB.Rand1, div));
            float u2 = rand2(float2(div, g_sceneCB.Rand2));
            float3 localHemisphereRay = UniformHemisphereSample(u1, u2);
            // Transform the ray direction to world space
            float3 worldHemisphereRay = mul(localHemisphereRay, basis);
            ray.Direction = normalize(worldHemisphereRay);
            TraceRay(Scene, RAY_FLAG_NONE, ~0, 0, 0, 0, ray, payload);
        
            if (payload.primitiveIndex == -1)
                continue;
        
            uint clusterIndex = gVoxelAssignmentBuffer[payload.primitiveIndex];
        
            if (clusterIndex == -1)
                continue;
        
            // Add the cluster only if it is not already in the list
            uint isNewCluster = 1 - (uint) alreadyOccupiedVoxel(clusterIndex);

        
            if (g_sceneCB.CurrentPhase == 1 && isNewCluster == 1)
            {
                uint2 faceClusterVisibility = gFaceClusterVisibility[voxelIndex * 6 + faceIdx];
                gVisibleClustersBuffer[faceClusterVisibility.x + visibleClusterCount] = clusterIndex;
            }
        
            visibleClusterCount += isNewCluster;
        }
    }

    if (g_sceneCB.CurrentPhase == 0)
    {
        uint startIndex = 0;
        InterlockedAdd(gClusterCount[0], visibleClusterCount, startIndex);
        gFaceClusterVisibility[voxelIndex * 6 + faceIdx] = uint2(startIndex, visibleClusterCount);
    }
}

[shader("closesthit")]
void MyClosestHitShader(inout RayPayload payload, in AABBAttributes attr)
{
    uint idx = gAABBVoxelIndices[attr.aabbIndex];;
    
    payload.primitiveIndex = idx;
}

[shader("intersection")]
void MyIntersectionShader()
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
        AABBAttributes attr;
        attr.aabbIndex = idx;
        
        ReportHit(tMin, 0, attr); // Report the hit with hitT as tMin
    }
}

[shader("miss")]
void MyMissShader(inout RayPayload payload)
{
    float4 background = float4(0.0f, 0.0f, 0.0f, 0.0f);
    payload.primitiveIndex = -1;
}

#endif // RAYTRACING_HLSL