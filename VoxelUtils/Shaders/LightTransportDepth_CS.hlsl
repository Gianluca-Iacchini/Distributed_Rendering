#include "../../VoxelUtils/Shaders/VoxelUtils.hlsli"

ConstantBuffer<ConstantBufferVoxelCommons> cbVoxelCommons : register(b0);
ConstantBuffer<ConstantBufferFrustumCulling> cbFrustumCulling : register(b1);
ConstantBuffer<Light> cbCamera : register(b2);

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space0);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space0);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space0);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space0);


Texture2D gCameraDepth : register(t0, space1);


RWByteAddressBuffer gVisibleFacesCounter : register(u0);
RWStructuredBuffer<uint> gIndirectLightVisibleFacesIndices : register(u1);
RWStructuredBuffer<uint> gGaussianVisibleFacesIndices : register(u2);
RWByteAddressBuffer gIndirectLightUpdatedVoxelsBitmap : register(u3);
RWByteAddressBuffer gGaussianUpdatedVoxelsBitmap : register(u4);
RWStructuredBuffer<uint3> gGaussianDispatchIndirectBuffer : register(u5);
RWStructuredBuffer<uint3> gIndirectLightDispatchIndirectBuffer : register(u6);

static const float SQRT_2 = 3.0f; //1.41421356237f;


[numthreads(128, 1, 1)]
void CS(uint3 DTid : SV_DispatchThreadID)
{
    uint threadLinearIndex = DTid.x;
    
    if (cbFrustumCulling.CurrentStep == 0)
    {
        
        if (threadLinearIndex >= ceil(cbFrustumCulling.FaceCount / 2.0f))
            return;
        
        gIndirectLightVisibleFacesIndices[threadLinearIndex] = UINT_MAX;
        gGaussianVisibleFacesIndices[threadLinearIndex] = UINT_MAX;
        
        if (threadLinearIndex >= cbFrustumCulling.VoxelCount)
            return;
        
        if (cbFrustumCulling.ResetRadianceBuffers == 1)
        {
            uint idx = threadLinearIndex >> 5u;
            idx = idx * 4;
            gIndirectLightUpdatedVoxelsBitmap.Store(idx, 0);
            gGaussianUpdatedVoxelsBitmap.Store(idx, 0);
        }
        
        if (threadLinearIndex == 0)
        {
            gVisibleFacesCounter.Store(0, 0);
            gVisibleFacesCounter.Store(4, 0);
            
            gGaussianDispatchIndirectBuffer[0] = uint3(0, 1, 1);
            gIndirectLightDispatchIndirectBuffer[0] = uint3(0, 1, 1);
        }

    }
    else if (cbFrustumCulling.CurrentStep == 1)
    {
        if (threadLinearIndex >= cbFrustumCulling.VoxelCount)
            return;

        bool isAtEdgeOfCamera = false;
        
        float margin = 20.0f;

        uint voxelLinearPoos = gVoxelHashedCompactBuffer[threadLinearIndex];
        float3 voxelPos = float3(GetVoxelPosition(voxelLinearPoos, cbVoxelCommons.voxelTextureDimensions));
        
        for (uint p = 0; p < 6; p++)
        {
            // Get the normal and distance (D) from the plane
            float4 frustumPlane = cbFrustumCulling.FrustumPlanes[p];
        
            float3 normal = frustumPlane.xyz;
            float d = frustumPlane.w;
            
            float3 aabbMin = voxelPos - float3(0.5f, 0.5f, 0.5f);
            float3 aabbMax = voxelPos + float3(0.5f, 0.5f, 0.5f);
        
            // Compute the positive vertex of the AABB for this plane
            float3 positiveVertex = float3(
            (normal.x > 0) ? aabbMax.x : aabbMin.x,
            (normal.y > 0) ? aabbMax.y : aabbMin.y,
            (normal.z > 0) ? aabbMax.z : aabbMin.z);
        
            float aabbDistance = dot(normal, positiveVertex) + d;
            
            // If the positive vertex is outside the plane, the AABB is outside the frustum
            if (aabbDistance + margin < 0)
            {
                return;
            }
            
            isAtEdgeOfCamera = isAtEdgeOfCamera | (aabbDistance < 0);
        }
        
        

        bool lit = false;
    
        float3 worldPos = mul(float4(voxelPos, 1.0f), cbVoxelCommons.VoxelToWorld).xyz;
		
        float3 shadowTestPoints[12];
    
        float3 offset = cbVoxelCommons.voxelCellSize;
        offset *= 3.0f;
    
        // Generate the 6 corner points of the voxel
        {
	        // Corners
            shadowTestPoints[0] = worldPos + float3(offset.x, 0.0, 0.0);
            shadowTestPoints[1] = worldPos + float3(-1.0 * offset.x, 0.0, 0.0);
            shadowTestPoints[2] = worldPos + float3(0.0, offset.y, 0.0);
            shadowTestPoints[3] = worldPos + float3(0.0, -1.0 * offset.y, 0.0);
            shadowTestPoints[4] = worldPos + float3(0.0, 0.0, offset.z);
            shadowTestPoints[5] = worldPos + float3(0.0, 0.0, -1.0 * offset.z);
        }
    
        float4 shadowCoord;
        float depth;
        for (uint i = 0; i < 6; ++i)
        {
            shadowCoord = mul(float4(shadowTestPoints[i], 1.0f), cbCamera.shadowMatrix);
            shadowCoord /= shadowCoord.w;
            depth = gCameraDepth.SampleCmpLevelZero(gShadowSampler, shadowCoord.xy, shadowCoord.z).r;
            if (depth > 0.0f)
            {
                lit = true;
                break;
            }
        }

        // Test the mid edge points
        {
	        // Edge mid points
            shadowTestPoints[0] = worldPos + float3(0.0, offset.y, -1.0 * offset.z);
            shadowTestPoints[1] = worldPos + float3(-1.0 * offset.x, 0.0, -1.0 * offset.z);
            shadowTestPoints[2] = worldPos + float3(offset.x, 0.0, -1.0 * offset.z);
            shadowTestPoints[3] = worldPos + float3(0.0, -1.0 * offset.y, -1.0 * offset.z);
            shadowTestPoints[4] = worldPos + float3(-1.0 * offset.x, offset.y, 0.0);
            shadowTestPoints[5] = worldPos + float3(offset.x, offset.y, 0.0);
            shadowTestPoints[6] = worldPos + float3(-1.0 * offset.x, -1.0 * offset.y, 0.0);
            shadowTestPoints[7] = worldPos + float3(offset.x, -1.0 * offset.y, 0.0);
            shadowTestPoints[8] = worldPos + float3(0.0, offset.y, offset.z);
            shadowTestPoints[9] = worldPos + float3(offset.x, 0.0, offset.z);
            shadowTestPoints[10] = worldPos + float3(0.0, -1.0 * offset.y, offset.z);
            shadowTestPoints[11] = worldPos + float3(-1.0 * offset.x, 0.0, offset.z);
        }
		
        if (!lit)
        {
            for (i = 0; i < 12; ++i)
            {
                shadowCoord = mul(float4(shadowTestPoints[i], 1.0f), cbCamera.shadowMatrix);
                shadowCoord /= shadowCoord.w;
                depth = gCameraDepth.SampleCmpLevelZero(gShadowSampler, shadowCoord.xy, shadowCoord.z).r;
        
                if (depth > 0.0f)
                {
                    lit = true;
                    break;
                }
            }
        }

        if (!lit)
        {
            return;
        }

            
        bool wasSet = false;
        
#ifndef GAUSSIAN_ONLY
        uint nIndirectLightFaces = 0;
        uint indirectLightStartAddress = 0;

        if (!IsVoxelPresent(threadLinearIndex, gIndirectLightUpdatedVoxelsBitmap))
        {
            nIndirectLightFaces += 6;
        }
            

        gVisibleFacesCounter.InterlockedAdd(0, nIndirectLightFaces, indirectLightStartAddress);

        wasSet = SetVoxelPresence(threadLinearIndex, gIndirectLightUpdatedVoxelsBitmap);
        if (!wasSet)
        {
            gIndirectLightVisibleFacesIndices[indirectLightStartAddress + 0] = threadLinearIndex * 6 + 0;
            gIndirectLightVisibleFacesIndices[indirectLightStartAddress + 1] = threadLinearIndex * 6 + 1;
            gIndirectLightVisibleFacesIndices[indirectLightStartAddress + 2] = threadLinearIndex * 6 + 2;
            gIndirectLightVisibleFacesIndices[indirectLightStartAddress + 3] = threadLinearIndex * 6 + 3;
            gIndirectLightVisibleFacesIndices[indirectLightStartAddress + 4] = threadLinearIndex * 6 + 4;
            gIndirectLightVisibleFacesIndices[indirectLightStartAddress + 5] = threadLinearIndex * 6 + 5;
        }
            
        uint buffer0Max = (uint) ceil((indirectLightStartAddress + nIndirectLightFaces));

        InterlockedMax(gIndirectLightDispatchIndirectBuffer[0].x, buffer0Max);

 #endif
        
        uint nGaussianFaces = 0;
        uint gaussianStartAddress = 0;
            
        if (!IsVoxelPresent(threadLinearIndex, gGaussianUpdatedVoxelsBitmap))
        {
            nGaussianFaces += 6;
        }
        gVisibleFacesCounter.InterlockedAdd(4, nGaussianFaces, gaussianStartAddress);
            
        
        wasSet = IsVoxelPresent(threadLinearIndex, gGaussianUpdatedVoxelsBitmap);
        if (!wasSet)
        {
            gGaussianVisibleFacesIndices[gaussianStartAddress + 0] = threadLinearIndex * 6 + 0;
            gGaussianVisibleFacesIndices[gaussianStartAddress + 1] = threadLinearIndex * 6 + 1;
            gGaussianVisibleFacesIndices[gaussianStartAddress + 2] = threadLinearIndex * 6 + 2;
            gGaussianVisibleFacesIndices[gaussianStartAddress + 3] = threadLinearIndex * 6 + 3;
            gGaussianVisibleFacesIndices[gaussianStartAddress + 4] = threadLinearIndex * 6 + 4;
            gGaussianVisibleFacesIndices[gaussianStartAddress + 5] = threadLinearIndex * 6 + 5;
        }
                
        uint buffer1Max = (uint) ceil((gaussianStartAddress + nGaussianFaces) / (128.0f));
        
        InterlockedMax(gGaussianDispatchIndirectBuffer[0].x, buffer1Max);
    }
}



