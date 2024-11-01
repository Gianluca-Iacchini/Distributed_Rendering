#include "VoxelUtils.hlsli"
#include "RaytracingUtils.hlsli"

struct ConstantBufferCamera
{
    XMFLOAT4X4 view;
	XMFLOAT4X4 invView;
	XMFLOAT4X4 projection;
	XMFLOAT4X4 invProjection;
	XMFLOAT4X4 viewProjection;
	XMFLOAT4X4 invViewProjection;
    
    // Use by shadow cameras only.
    XMFLOAT3 eyePosition;
    float nearPlane;

    float farPlane;
    float _pad0;
    float _pad1;
    float _pad2;
};

ConstantBuffer<ConstantBufferVoxelCommons> cbVoxelCommons : register(b0);
ConstantBuffer<ConstantBufferFrustumCulling> cbFrustumCulling : register(b1);
ConstantBuffer<ConstantBufferCamera> cbCamera : register(b2);

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space0);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space0);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space0);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space0);

StructuredBuffer<AABB> gVoxelAABBBuffer : register(t0, space1);
StructuredBuffer<ClusterAABBInfo> gClusterAABBInfoBuffer : register(t1, space1);
// Map from aabbVoxelIndices to gVoxelIndicesCompactBuffer.
StructuredBuffer<uint> gAABBVoxelIndices : register(t2, space1);

RaytracingAccelerationStructure Scene : register(t0, space2);

RWByteAddressBuffer gVisibleFaceCounter : register(u0);
RWStructuredBuffer<uint> gVisibleFaceIndices : register(u1);
RWStructuredBuffer<uint3> gDispatchIndirectBuffer : register(u2);

RWStructuredBuffer<uint2> gFaceRadianceBuffer : register(u0, space1);
RWStructuredBuffer<uint2> gFaceFilteredRadianceBuffer : register(u1, space1);

static const float SQRT_2 = 1.41421356237f;

groupshared uint gsIsOutsideFrustum;
groupshared uint gsIsOccluded;

groupshared uint gsMaxFaces;

[numthreads(128, 1, 1)]
void CS(uint3 DTid : SV_DispatchThreadID, uint3 groupId : SV_GroupID, uint3 threadIdx : SV_GroupThreadID)
{
    uint threadLinearIndex = DTid.x;
    
    if (cbFrustumCulling.CurrentStep == 0)
    {
        
        if (threadLinearIndex >= cbFrustumCulling.FaceCount)
            return;
        
        
        gFaceRadianceBuffer[threadLinearIndex] = uint2(0, 0);
        gFaceFilteredRadianceBuffer[threadLinearIndex] = uint2(0, 0);

        if (threadLinearIndex < cbFrustumCulling.FaceCount / 2)
        {
            gVisibleFaceIndices[threadLinearIndex] = UINT_MAX;
        }
        
        if (threadLinearIndex >= cbFrustumCulling.VoxelCount)
            return;
        
        if (threadLinearIndex == 0)
        {
            gVisibleFaceCounter.Store(0, 0);
            gDispatchIndirectBuffer[0] = uint3(0, 1, 1);
            gDispatchIndirectBuffer[1] = uint3(0, 1, 1);
        }

    }

    else if (cbFrustumCulling.CurrentStep == 1)
    {
        if (groupId.x >= cbFrustumCulling.AABBGroupCount)
            return;
        

        
        ClusterAABBInfo aabbInfo = gClusterAABBInfoBuffer[groupId.x];
        uint aabbsPerThread = ceil((float) aabbInfo.ClusterElementCount / 128.0f);
        uint aabbStart = aabbInfo.ClusterStartIndex + threadIdx.x * aabbsPerThread;
        uint aabbEnd = min(aabbStart + aabbsPerThread, aabbInfo.ClusterStartIndex + aabbInfo.ClusterElementCount);
        

        if (threadIdx.x == 0)
        {
            gsIsOutsideFrustum = 0;
            gsIsOccluded = 1;
            gsMaxFaces = 0;
        }
        
        GroupMemoryBarrierWithGroupSync();
        
        if (threadIdx.x < 6)
        {

            // Get the normal and distance (D) from the plane
            float4 frustumPlane = cbFrustumCulling.FrustumPlanes[threadIdx.x];
        
            float3 normal = frustumPlane.xyz;
            float d = frustumPlane.w;
            
            float3 aabbMin = aabbInfo.Min;
            float3 aabbMax = aabbInfo.Max;
        
            // Compute the positive vertex of the AABB for this plane
            float3 positiveVertex = float3(
            (normal.x > 0) ? aabbMax.x : aabbMin.x,
            (normal.y > 0) ? aabbMax.y : aabbMin.y,
            (normal.z > 0) ? aabbMax.z : aabbMin.z);
        

            // If the positive vertex is outside the plane, the AABB is outside the frustum
            if (dot(normal, positiveVertex) + d < 0)
            {
                InterlockedOr(gsIsOutsideFrustum, 1);
            }
        }
        
        GroupMemoryBarrierWithGroupSync();
        
        float3 eyePos = mul(float4(cbCamera.eyePosition, 1.0f), cbVoxelCommons.WorldToVoxel).xyz;
        
        if (gsIsOutsideFrustum == 0)
        {
            float3 facesAndCorners[14] =
            {
                // 4 edge midpoints
                float3(0.0f, 0.0f, -2.0f),
                float3(0.0f, 0.0f, 2.0f),
                float3(0.0f, -2.0f, 0.0f),
                float3(0.0f, 2.0f, 0.0f),
                float3(-2.0f, 0.0f, 0.0f),
                float3(2.0f, 0.0f, 0.0f),
                
                float3(-SQRT_2, -SQRT_2, -SQRT_2),
                float3(-SQRT_2, -SQRT_2, SQRT_2),
                float3(-SQRT_2, SQRT_2, -SQRT_2),
                float3(-SQRT_2, SQRT_2, SQRT_2),
                float3(SQRT_2, -SQRT_2, -SQRT_2),
                float3(SQRT_2, -SQRT_2, SQRT_2),
                float3(SQRT_2, SQRT_2, -SQRT_2),
                float3(SQRT_2, SQRT_2, SQRT_2)
            };

            RayQuery < RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_TRIANGLES > q;
            

            bool isOccluded = (gsIsOccluded == 1);
            
            for (uint v = aabbStart; v < aabbEnd && isOccluded; v++)
            {

                AABB voxelAABB = gVoxelAABBBuffer[v];
                float3 aabbCenter = (voxelAABB.Min + voxelAABB.Max) * 0.5f;
                
                for (uint f = 0; f < 14 && isOccluded; f++)
                {
                    isOccluded = (gsIsOccluded == 1);

                    bool hitSomething = false;
                    float3 origin = aabbCenter + facesAndCorners[f];
                    float3 direction = eyePos - origin;
                    float distance = length(direction);
                
                    RayDesc ray;
                    ray.Origin = origin;
                    ray.Direction = direction / distance;
                    ray.TMin = 0.01f;
                    ray.TMax = distance;
                
                    q.TraceRayInline(Scene, RAY_FLAG_NONE, ~0, ray);
                    
                    while (q.Proceed())
                    {
                        if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
                        {
                            q.CommitProceduralPrimitiveHit(1.0f);
                            hitSomething = true;
                            break;
                        }
                        
                    }
                    
                    if (!hitSomething)
                    {
                        isOccluded = false;
                        gsIsOccluded = 0;
                    }
                }
            }
        }
        
        GroupMemoryBarrierWithGroupSync();
            
        if (gsIsOccluded == 0 && (aabbEnd > aabbStart))
        {
            uint nFaces = (aabbEnd - aabbStart) * 3;

            uint startAddress = 0;
            gVisibleFaceCounter.InterlockedAdd(0, nFaces, startAddress);

            uint3 faceIdx = uint3(2, 4, 0);
            
            for (uint i = aabbStart; i < aabbEnd; i++)
            {
                uint voxIdx = gAABBVoxelIndices[i];
                
                AABB voxAABB = gVoxelAABBBuffer[i];
                float3 center = (voxAABB.Min + voxAABB.Max) * 0.5f;
                
                faceIdx.x += step(center.x, eyePos.x);
                faceIdx.y += step(center.y, eyePos.y);
                faceIdx.z += step(center.z, eyePos.z);
                
                // Up to 3 faces are visible at a time
                gVisibleFaceIndices[startAddress + (i - aabbStart) * 3 + 0] = voxIdx * 6 + faceIdx.x;
                gVisibleFaceIndices[startAddress + (i - aabbStart) * 3 + 1] = voxIdx * 6 + faceIdx.y;
                gVisibleFaceIndices[startAddress + (i - aabbStart) * 3 + 2] = voxIdx * 6 + faceIdx.z;
            }



            uint buffer0Max = startAddress + nFaces;
            uint buffer1Max = (uint) ceil((buffer0Max) / 128.0f);
            InterlockedMax(gDispatchIndirectBuffer[0].x, buffer0Max);
            InterlockedMax(gDispatchIndirectBuffer[1].x, buffer1Max);


        }
    }
}



