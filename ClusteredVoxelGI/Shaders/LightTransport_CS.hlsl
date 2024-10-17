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

RWByteAddressBuffer gVisibleVoxels : register(u0);
RWByteAddressBuffer gVisibleVoxelCounter : register(u1);
RWStructuredBuffer<uint> gVisibleVoxelIndices : register(u2);
RWStructuredBuffer<uint3> gDispatchIndirectBuffer : register(u3);

RWStructuredBuffer<float> gVoxelRadiance : register(u0, space1);

static const float SQRT_2 = 1.41421356237f;

groupshared uint gsIsOutsideFrustum;
groupshared uint gsIsOccluded;
groupshared uint gsStartAddress;

[numthreads(128, 1, 1)]
void CS(uint3 DTid : SV_DispatchThreadID, uint3 groupId : SV_GroupID, uint3 threadIdx : SV_GroupThreadID)
{
    uint threadLinearIndex = DTid.x;
    
    if (cbFrustumCulling.CurrentStep == 0)
    {
        if (threadLinearIndex >= cbFrustumCulling.VoxelCount)
            return;
        
        if (threadLinearIndex == 0)
        {
            gVisibleVoxelCounter.Store(0, 0);
            gDispatchIndirectBuffer[0] = uint3(0, 1, 1);
        }
        
        gVoxelRadiance[threadLinearIndex] = 0.0f;
        
        uint bit = threadLinearIndex & 31u;
        
        //if (bit == 0)
        //{
        uint index = threadLinearIndex >> 5u;
        index = index * 4;
        uint outval;
        gVisibleVoxels.InterlockedAnd(index, (1u << bit), outval);
        //}

        
        gVisibleVoxelIndices[threadLinearIndex] = UINT_MAX;
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
            gsStartAddress = 0;
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
            
           
            float3 eyePos = mul(float4(cbCamera.eyePosition, 1.0f), cbVoxelCommons.WorldToVoxel).xyz;

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
                    ray.TMin = 0.1f;
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
            
        if (gsIsOccluded == 0)
        {
            if (threadIdx.x == 0)
            {
                gVisibleVoxelCounter.InterlockedAdd(0, aabbInfo.ClusterElementCount, gsStartAddress);
            }

            GroupMemoryBarrierWithGroupSync();
    
            uint startAddr = gsStartAddress;
                
            for (uint i = aabbStart; i < aabbEnd; i++)
            {
                gVisibleVoxelIndices[startAddr + i - aabbInfo.ClusterStartIndex] = gAABBVoxelIndices[i];
            }
        
            // We will use this buffer to dispatch the indirect draw call with a number of thread groups equal to
            // visibileVoxels / 128.
               
            if (threadIdx.x == 0)
            {
                InterlockedMax(gDispatchIndirectBuffer[0].x, ceil((startAddr + aabbInfo.ClusterElementCount) / 128.0f));
            }

        }
    }
}



