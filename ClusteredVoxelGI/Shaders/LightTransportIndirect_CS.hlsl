#include "VoxelUtils.hlsli"

ConstantBuffer<ConstantBufferVoxelCommons> cbVoxelCommons : register(b0);
ConstantBuffer<ConstantBufferIndirectLightTransport> cbIndirectLight : register(b1);

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space0);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space0);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space0);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space0);

StructuredBuffer<ClusterData> gClusterDataBuffer : register(t0, space1);
StructuredBuffer<uint> gVoxelsInCluster : register(t1, space1);
StructuredBuffer<uint> gVoxelAssignmentMap : register(t2, space1);
StructuredBuffer<float3> gVoxelColorBuffer : register(t3, space1);
StructuredBuffer<float3> gVoxelNormalBuffer: register(t4, space1);

StructuredBuffer<uint2> gVoxelFaceDataBuffer : register(t0, space2);
// The element i contains the start index in gVoxelFaceDataBuffer and the number of the faces for the voxel with index i
StructuredBuffer<uint2> gVoxelFaceStartCountBuffer : register(t1, space2);

StructuredBuffer<AABB> gVoxelAABBBuffer : register(t0, space3);
StructuredBuffer<ClusterAABBInfo> gClusterAABBInfoBuffer : register(t1, space3);
// Map from aabbVoxelIndices to gVoxelIndicesCompactBuffer.
StructuredBuffer<uint> gAABBVoxelIndices : register(t2, space3);

StructuredBuffer<uint2> gFaceClusterVisibility : register(t0, space4);
// Stores all the visible clusters for all the faces. Clusters visible from the same faced are stored in sequence.
StructuredBuffer<uint> gVisibleClustersBuffer : register(t1, space4);

StructuredBuffer<float> gFaceClusterPenaltyBuffer : register(t0, space5);
StructuredBuffer<float> gFaceCloseVoxelsPenaltyBuffer : register(t1, space5);

StructuredBuffer<float3> gLitVoxels : register(t0, space6);
StructuredBuffer<uint4> gLitClusters : register(t1, space6);

ByteAddressBuffer gVisibleFaceCounter : register(t0, space7);
StructuredBuffer<uint> gVisibleFaceIndices : register(t1, space7);

RWStructuredBuffer<uint2> gFaceRadianceBuffer : register(u0);
RWStructuredBuffer<uint2> gFaceFilteredRadianceBuffer : register(u1, space1);

groupshared uint3 gsRadiance;

float3 clusterToVoxelIrradiancePerVoxelArrayVoxel(ClusterData cData, uint voxelIndex, float3 voxelWorldCoords, float3 voxelNormal)
{
    bool coplanar;
    uint3 voxelCoordDifference;
    float3 litVoxelWorldCoords;
    float formFactorVoxelIrradianceField;
    float formFactor;
    float3 currentVoxelToLitVoxel;
    float3 irradianceEmitter;

    float3 irradianceAccumulated = float3(0.0, 0.0, 0.0);
    float3 emitterNormal = cbIndirectLight.LightDirection;
    float3 emitterPosition = cbIndirectLight.LightPosition;
    float emitterRadiance = 25.0f; //cbIndirectLight.LightIntensity;
    float attenuationFactor = 0.01f;
    
    uint3 voxelTexCoords = GetVoxelPosition(gVoxelHashedCompactBuffer[voxelIndex], cbVoxelCommons.voxelTextureDimensions);
    
    uint3 litVoxelTexCoords;
    
    
    for (uint i = cData.FirstDataIndex; i < cData.FirstDataIndex + cData.VoxelCount; i++)
    {
        uint voxelIdx = gVoxelsInCluster[i];
        
        // If the voxel we are checking is the same as the voxel we are calculating the irradiance for, skip it
        if (voxelIdx == voxelIndex)
        {
            continue;
        }
        
        // Only lit voxels are considered
        if (any(gLitVoxels[voxelIdx] > 0.0f))
        {
            litVoxelTexCoords = GetVoxelPosition(gVoxelHashedCompactBuffer[voxelIdx], cbVoxelCommons.voxelTextureDimensions);
            litVoxelWorldCoords = mul(float4(float3(litVoxelTexCoords), 1.0f), cbVoxelCommons.VoxelToWorld).xyz;
            

            currentVoxelToLitVoxel = litVoxelWorldCoords - voxelWorldCoords;
            float distanceToVoxel = length(currentVoxelToLitVoxel);
            currentVoxelToLitVoxel = currentVoxelToLitVoxel / distanceToVoxel;
            
            float cosineAngle = dot(-1.0 * currentVoxelToLitVoxel, cData.Normal);
            
            if ((distanceToVoxel < MAX_DISTANCE_VOXEL_OFFSET) &&
                (cosineAngle > 0.0f) && (cosineAngle < MINIMUM_ANGLE_VOXEL_OFFSET))
            {
                float3 offsetValue = cbVoxelCommons.voxelCellSize * 1.5f;
                
                float3 tempPosition0 = litVoxelWorldCoords + cData.Normal * offsetValue;
                float3 tempPosition1 = litVoxelWorldCoords - cData.Normal * offsetValue;
                
                if (distanceSq(voxelWorldCoords, tempPosition0) > distanceSq(voxelWorldCoords, tempPosition1))
                {
                    litVoxelWorldCoords = tempPosition0;
                }
                else
                {
                    litVoxelWorldCoords = tempPosition1;
                }
            }
            
            currentVoxelToLitVoxel = normalize(litVoxelWorldCoords - voxelWorldCoords);
            voxelCoordDifference = voxelTexCoords - litVoxelTexCoords;
            coplanar = any(voxelCoordDifference == 0);
            
            if (coplanar && (distanceToVoxel < MAX_DISTANCE_VOXEL_OFFSET) &&
                (dot(voxelNormal, cData.Normal) < 1.0f) && (dot(voxelNormal, cData.Normal) >= 0.0f))
            {
                float3 tempPos0 = litVoxelWorldCoords + cData.Normal * cbVoxelCommons.voxelCellSize * 1.5f;
                float3 tempPos1 = litVoxelWorldCoords - cData.Normal * cbVoxelCommons.voxelCellSize * 1.5f;
                float3 direction0 = tempPos0 - voxelWorldCoords;
                float3 direction1 = tempPos1 - voxelWorldCoords;
                
                if (dot(direction0, cData.Normal) < dot(direction1, cData.Normal))
                {
                    litVoxelWorldCoords = tempPos0;
                }
                else
                {
                    litVoxelWorldCoords = tempPos1;
                }
            }

            formFactor = differentialAreaFormFactor(cData.Normal, emitterNormal);
            irradianceEmitter = formFactor * cData.Color * emitterRadiance;
            currentVoxelToLitVoxel = normalize(litVoxelWorldCoords - voxelWorldCoords);
            formFactorVoxelIrradianceField = differentialAreaFormFactor(currentVoxelToLitVoxel, voxelWorldCoords, cData.Normal, litVoxelWorldCoords, 3.0f);
            irradianceAccumulated += irradianceEmitter * formFactorVoxelIrradianceField;
        }
        
    }

    return irradianceAccumulated * attenuationFactor;
}


float3 gatherIrradianceFromNeighbour(uint clusterIdx)
{
    ClusterData cData = gClusterDataBuffer[clusterIdx];
    
    float3 clusterWorldPos = mul(float4(cData.Center.x, cData.Center.y, cData.Center.z, 1.0f), cbVoxelCommons.VoxelToWorld).xyz;
    
    int numberNeighbour = cData.NeighbourCount;
    float3 accumulatedIrradiance = float3(0.0f, 0.0f, 0.0f);


    for (int i = 0; i < numberNeighbour; ++i)
    {
        int neighbourIndex = cData.ClusterNeighbours[i];
        float3 mainDirectionNeighbour = gClusterDataBuffer[neighbourIndex].Normal;
        uint3 centerAABB = gClusterDataBuffer[neighbourIndex].Center;
        float3 neighbourWorldCoords = mul(float4(float3(centerAABB), 1.0f), cbVoxelCommons.VoxelToWorld).xyz;
        float3 neighbourToVoxel = normalize(neighbourWorldCoords - clusterWorldPos);

		// Only consider same main direction clusters.
        if ((dot(cData.Normal, neighbourToVoxel) > 0.0) && (gLitClusters[neighbourIndex].w > 0))
        {
            float distanceFromVoxelSq = distanceSq(neighbourWorldCoords, clusterWorldPos);

            if (distanceFromVoxelSq > 25.0f)
            {
                uint3 neighIrradiance = gLitClusters[neighbourIndex].xyz;
                float3 neighbourIrradiance = float3(neighIrradiance) / IRRADIANCE_FIELD_MULTIPLIER;
                float formFactor = differentialAreaFormFactor(neighbourToVoxel, clusterWorldPos, mainDirectionNeighbour, neighbourWorldCoords, 14860);
                accumulatedIrradiance += formFactor * neighbourIrradiance;
            }
        }
    }

    return accumulatedIrradiance;
}

[numthreads(1, 128, 1)]
void CS( uint3 DTid : SV_DispatchThreadID, uint3 threadGroupId : SV_GroupThreadID)
{
    uint threadID = DTid.x;
    
    uint nVisibleVoxels = gVisibleFaceCounter.Load(0);
    
    if (threadGroupId.y == 0)
    {
        gsRadiance = uint3(0, 0, 0);
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    if (threadID > nVisibleVoxels)
        return;
    
    uint faceIdx = gVisibleFaceIndices[threadID];
    
    uint2 faceData = gVoxelFaceDataBuffer[faceIdx];
    
    uint voxelIndex = faceData.x;
    

    float3 voxelWorldPos = float3(GetVoxelPosition(gVoxelHashedCompactBuffer[voxelIndex], cbVoxelCommons.voxelTextureDimensions));
    voxelWorldPos = mul(float4(voxelWorldPos, 1.0f), cbVoxelCommons.VoxelToWorld).xyz;
    
    float3 radiance = 0.0f;
    
    float3 voxelNormal = gVoxelNormalBuffer[voxelIndex];
    

    uint2 clusterStartCount = gFaceClusterVisibility[faceIdx];

    
    uint clusterPerThread = ceil((float) clusterStartCount.y / 128.0f);
    uint clusterStart = clusterStartCount.x + threadGroupId.y * clusterPerThread;
    uint clusterEnd = min(clusterStart + clusterPerThread, clusterStartCount.x + clusterStartCount.y);
    
    // Max cluster number per voxel face is 128.
    float3 currRadiance[4];
    uint nIteration = 0;
    
    for (uint visibleClusterIdx = clusterStart; visibleClusterIdx < clusterEnd; visibleClusterIdx++)
    {
        uint clusterIdx = gVisibleClustersBuffer[visibleClusterIdx];

        if (gLitClusters[clusterIdx].w > 0)
        {
            ClusterData clusterData = gClusterDataBuffer[clusterIdx];
            float3 clusterWorldPos = mul(float4(clusterData.Center, 1.0f), cbVoxelCommons.VoxelToWorld).xyz;
                
            float3 voxelToCluster = clusterWorldPos - voxelWorldPos;
            float distance = length(voxelToCluster);
                
            uint4 clusterRadianceUint = gLitClusters[clusterIdx];
            float3 clusterRadiance = float3(clusterRadianceUint.xyz) / IRRADIANCE_FIELD_MULTIPLIER;
            


            if (distance > 5.0f)
            {
                float formFactor = differentialAreaFormFactor(voxelToCluster / distance, voxelWorldPos, clusterData.Normal, clusterWorldPos, 14860);
                currRadiance[nIteration] = clusterRadiance * formFactor;
                radiance += currRadiance[nIteration];
            }
            else
            {
                float penaltyFactor = gFaceClusterPenaltyBuffer[faceIdx];
                currRadiance[nIteration] = clusterToVoxelIrradiancePerVoxelArrayVoxel(clusterData, voxelIndex, voxelWorldPos, voxelNormal);
                
                radiance += currRadiance[nIteration];
                radiance /= penaltyFactor;
            }

            nIteration++;
        }
    }

    uint3 uintRadiance = uint3(radiance * IRRADIANCE_FIELD_MULTIPLIER);
    InterlockedAdd(gsRadiance.x, uintRadiance.x);
    InterlockedAdd(gsRadiance.y, uintRadiance.y);
    InterlockedAdd(gsRadiance.z, uintRadiance.z);
    
    GroupMemoryBarrierWithGroupSync();
    

    //nIteration = 0;
    //float3 neighbourRadiance = float3(0.0f, 0.0f, 0.0f);
    //radiance = float3(gsRadiance) / IRRADIANCE_FIELD_MULTIPLIER;
    
    //GroupMemoryBarrierWithGroupSync();
    
    //bool addExtraIrradiace = false;
    
    //for (visibleClusterIdx = clusterStart; visibleClusterIdx < clusterEnd; visibleClusterIdx++)
    //{
    //    uint clusterIdx = gVisibleClustersBuffer[visibleClusterIdx];
    //    float3 weight = currRadiance[nIteration] / radiance;
        
    //    if (any(weight >= 0.5f))
    //    {
    //        neighbourRadiance += gatherIrradianceFromNeighbour(clusterIdx);
    //        addExtraIrradiace = true;
    //    }
        
    //    nIteration++;
    //}
    

    //if (addExtraIrradiace)
    //{
    //    uintRadiance = uint3(neighbourRadiance * IRRADIANCE_FIELD_MULTIPLIER);
    //    InterlockedAdd(gsRadiance.x, uintRadiance.x);
    //    InterlockedAdd(gsRadiance.y, uintRadiance.y);
    //    InterlockedAdd(gsRadiance.z, uintRadiance.z);
    //}

    //GroupMemoryBarrierWithGroupSync();
    
    if (threadGroupId.y == 0)
    {
        float3 finalRadiance = float3(gsRadiance) / IRRADIANCE_FIELD_MULTIPLIER;
        
        uint packedXY = PackFloats16(float2(finalRadiance.x, finalRadiance.y));
        uint packedZ = PackFloats16(float2(finalRadiance.z, 0.0f));
        
        gFaceRadianceBuffer[faceIdx] = uint2(packedXY, packedZ);
    }
}