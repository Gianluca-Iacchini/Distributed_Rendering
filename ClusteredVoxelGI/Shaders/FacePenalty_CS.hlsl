#include "VoxelUtils.hlsli"

ConstantBuffer<ConstantBufferVoxelCommons> cbVoxelCommons : register(b0);
ConstantBuffer<ConstantBufferIndirectLightTransport> cbLightTransport : register(b1);

ByteAddressBuffer gVoxelOccupiedBuffer : register(t0, space0);

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space1);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space1);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space1);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space1);

StructuredBuffer<ClusterData> gClusterDataBuffer : register(t0, space2);
StructuredBuffer<uint> gVoxelsInCluster : register(t1, space2);
StructuredBuffer<uint> gVoxelAssignmentMap : register(t2, space2);
StructuredBuffer<float3> gVoxelColorBuffer : register(t3, space2);
StructuredBuffer<float3> gVoxelNormalBuffer : register(t4, space2);

StructuredBuffer<uint2> gVoxelFaceDataBuffer : register(t0, space3);
StructuredBuffer<uint2> gVoxelFaceStartCountBuffer : register(t1, space3);

StructuredBuffer<uint2> gFaceClusterVisibility : register(t0, space4);
StructuredBuffer<uint> gVisibleClustersBuffer : register(t1, space4);

RWStructuredBuffer<float> gFaceClusterPenaltyBuffer : register(u0, space0);
RWStructuredBuffer<float> gFaceCloseVoxelsPenaltyBuffer : register(u1, space0);

uint2 FindHashedCompactedPositionIndex(uint3 coord, uint3 gridDimension)
{
    uint2 result = uint2(0, 0); // y field is control value, 0 means element not found, 1 means element found
    uint indirectionIndex = gridDimension.z * coord.z + coord.y;
    uint index = gIndirectionIndexBuffer[indirectionIndex];
    uint rank = gIndirectionRankBuffer[indirectionIndex];
    uint hashedPosition = GetLinearCoord(coord, gridDimension);
    
    if (rank == 0)
        return result;
    
    uint tempHashed;
    uint startIndex = index;
    uint endIndex = index + rank;
    uint currentIndex = (startIndex + endIndex) / 2;

    for (int i = 0; i < int(12); ++i)
    {
        tempHashed = gVoxelHashedCompactBuffer[currentIndex];

        if (tempHashed == hashedPosition)
        {
            return uint2(currentIndex, 1);
        }

        if (tempHashed < hashedPosition)
        {
            startIndex = currentIndex;
            currentIndex = (startIndex + endIndex) / 2;
        }
        else
        {
            endIndex = currentIndex;
            currentIndex = (startIndex + endIndex) / 2;
        }
    }

    return result;
}


float clusterToVoxelIrradiancePerVoxelArrayVoxel(ClusterData cData, uint voxelIndex, float3 voxelWorldCoords)
{
    bool coplanar;
    uint3 voxelCoordDifference;
    float3 litVoxelWorldCoords;
    float formFactorVoxelIrradianceField;
    float formFactor;
    float3 currentVoxelToLitVoxel;
    float irradianceEmitter;

    float irradianceAccumulated = 0.0f;
    float3 emitterNormal = cbLightTransport.LightDirection;
    float3 emitterPosition = cbLightTransport.LightPosition;
    float emitterRadiance = 15.0f; //cbLightTransport.LightIntensity;
    float attenuationFactor = 0.5f;
    
    uint3 voxelTexCoords = GetVoxelPosition(gVoxelHashedCompactBuffer[voxelIndex], cbVoxelCommons.voxelTextureDimensions);
    
    uint3 litVoxelTexCoords;
    
    float3 voxelNormal = gVoxelNormalBuffer[voxelIndex];
    
    for (uint i = cData.FirstDataIndex; i < cData.FirstDataIndex + cData.VoxelCount; i++)
    {
        uint voxelIdx = gVoxelsInCluster[i];
        
        // If the voxel we are checking is the same as the voxel we are calculating the irradiance for, skip it
        if (voxelIdx == voxelIndex)
        {
            continue;
        }

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
            
        irradianceEmitter = 1.0f;
        currentVoxelToLitVoxel = normalize(litVoxelWorldCoords - voxelWorldCoords);
        formFactorVoxelIrradianceField = differentialAreaFormFactor(currentVoxelToLitVoxel, voxelWorldCoords, cData.Normal, litVoxelWorldCoords, 3.0f);
        irradianceAccumulated += irradianceEmitter * formFactorVoxelIrradianceField;
    }
        
    

    return irradianceAccumulated * attenuationFactor;
}

float clusterToVoxelIrradiance(float3 voxelWorldCoord, ClusterData cData)
{
    float3 clusterWorldCoords = mul(float4(cData.Center, 1.0f), cbVoxelCommons.VoxelToWorld).xyz;
    float3 voxelToCluster = normalize(clusterWorldCoords - voxelWorldCoord);
    float formFactor = differentialAreaFormFactor(voxelToCluster, voxelWorldCoord, cData.Normal, clusterWorldCoords, CLUSTER_TO_VOXEL_FORM_FACTOR_ADD);

    // 1.0f = Max possible irradiance
    return formFactor * 1.0f;
}

float emulateIrradiance(uint2 visibleClustersStartCount, uint voxelIndex)
{
    float tempIrradiance = 0.0f;
    float3 voxelWorldCoords = mul(float4(GetVoxelPosition(gVoxelHashedCompactBuffer[voxelIndex], cbVoxelCommons.voxelTextureDimensions), 1.0f), cbVoxelCommons.VoxelToWorld).xyz;
    
    for (uint cl = visibleClustersStartCount.x; cl < visibleClustersStartCount.x + visibleClustersStartCount.y; cl++)
    {
        uint clusterIndex = gVisibleClustersBuffer[cl];
        
        ClusterData cData = gClusterDataBuffer[clusterIndex];
        
        float3 clusterWorldCoords = mul(float4(cData.Center, 1.0f), cbVoxelCommons.VoxelToWorld).xyz;
        float clusterToVoxelDistance = length(clusterWorldCoords - voxelWorldCoords);
        
        if (clusterToVoxelDistance > 5.0f)
        {
            tempIrradiance += clusterToVoxelIrradiance(voxelWorldCoords, cData);
        }
        else
        {
            tempIrradiance += clusterToVoxelIrradiancePerVoxelArrayVoxel(cData, voxelIndex, voxelWorldCoords);
        }
    }
    
    return tempIrradiance;
}

float computeNeighbourIrradiancePenalty(uint faceIdx, uint2 faceData)
{
    uint2 visibleClustersStartCount = gFaceClusterVisibility[faceIdx];
    
    float currentIrradiance = emulateIrradiance(visibleClustersStartCount, faceData.x);
    
    int3 crossDir0;
    int3 crossDir1;
    int3 faceDir;
    
    // 0: -z, 1: +z, 2: -x, 3: +x, 4: -y, 5: +y
    switch (faceData.y)
    {
        case 0:
            crossDir0 = int3(1, 0, 0);
            crossDir1 = int3(0, 1, 0);
            faceDir = int3(0, 0, -1);
            break;
        case 1:
             crossDir0 = int3(1, 0, 0);
            crossDir1 = int3(0, 1, 0);
            faceDir = int3(0, 0, 1);
            break;
        case 2:
            crossDir0 = int3(0, 1, 0);
            crossDir1 = int3(0, 0, 1);
            faceDir = int3(-1, 0, 0);
            break;
        case 3:
            crossDir0 = int3(0, 1, 0);
            crossDir1 = int3(0, 0, 1);
            faceDir = int3(1, 0, 0);
            break;
        case 4:
            crossDir0 = int3(1, 0, 0);
            crossDir1 = int3(0, 0, 1);
            faceDir = int3(0, -1, 0);
            break;
        case 5:
            crossDir0 = int3(1, 0, 0);
            crossDir1 = int3(0, 0, 1);
            faceDir = int3(0, 1, 0);
            break;
    }
    
    int3 iVoxelCoord = int3(GetVoxelPosition(gVoxelHashedCompactBuffer[faceData.x], cbVoxelCommons.voxelTextureDimensions));
    int3 neighbourCoords[17];
    int3 texCoordOffset = iVoxelCoord + faceDir;
    
    neighbourCoords[0] = iVoxelCoord - crossDir0;
    neighbourCoords[1] = iVoxelCoord + crossDir0;
    neighbourCoords[2] = iVoxelCoord - crossDir1;
    neighbourCoords[3] = iVoxelCoord + crossDir1;
    neighbourCoords[4] = iVoxelCoord - crossDir0 - crossDir1;
    neighbourCoords[5] = iVoxelCoord - crossDir0 + crossDir1;
    neighbourCoords[6] = iVoxelCoord + crossDir0 - crossDir1;
    neighbourCoords[7] = iVoxelCoord + crossDir0 + crossDir1;
    neighbourCoords[8] = texCoordOffset;
    neighbourCoords[9] = texCoordOffset - crossDir0;
    neighbourCoords[10] = texCoordOffset + crossDir0;
    neighbourCoords[11] = texCoordOffset - crossDir1;
    neighbourCoords[12] = texCoordOffset + crossDir1;
    neighbourCoords[13] = texCoordOffset - crossDir0 - crossDir1;
    neighbourCoords[14] = texCoordOffset - crossDir0 + crossDir1;
    neighbourCoords[15] = texCoordOffset + crossDir0 - crossDir1;
    neighbourCoords[16] = texCoordOffset + crossDir0 + crossDir1;
    
    float meanIrradiance = 0.0f;
    float counterNumIrradiance = 0.0f;
    
    for (uint i = 0; i < 17; i++)
    {
        int3 nCoord = neighbourCoords[i];
        if (any(nCoord < 0)|| any(nCoord > cbVoxelCommons.voxelTextureDimensions))
            continue;

        uint2 result = FindHashedCompactedPositionIndex(uint3(nCoord), cbVoxelCommons.voxelTextureDimensions);
        
        if (result.y != 0)
        {
            uint2 voxelFaceCount = gVoxelFaceStartCountBuffer[result.x];
            for (uint f = voxelFaceCount.x; f < voxelFaceCount.x + voxelFaceCount.y; f++)
            {
                if (gVoxelFaceDataBuffer[f].y == faceData.y)
                {
                    visibleClustersStartCount = gFaceClusterVisibility[f];
                    meanIrradiance += emulateIrradiance(visibleClustersStartCount, result.x);
                    counterNumIrradiance += 1.0f;
                    break;
                }
            }
        }
    }
    
    if (counterNumIrradiance < 0.5f)
    {
        return 1.0f;
    }
    
    meanIrradiance /= counterNumIrradiance;

	// Add penalty to those voxel faces that have high irradiance
    if (currentIrradiance > (meanIrradiance * 0.5f))
    {
        return (meanIrradiance / currentIrradiance);
    }
	

	// Help those voxel faces that have low irradiance
    if (currentIrradiance < meanIrradiance)
    {
        return (meanIrradiance / currentIrradiance);
    }

    return 1.0f;
}

float computeCloserVoxelVisibilityFacePenalty(uint faceIdx, uint2 faceData)
{
    uint2 visibleClustersStartCount = gFaceClusterVisibility[faceIdx];
    
    float3 voxelWorldCoords = mul(float4(GetVoxelPosition(gVoxelHashedCompactBuffer[faceData.x], cbVoxelCommons.voxelTextureDimensions), 1.0f), cbVoxelCommons.VoxelToWorld).xyz;

    float penaltyValue = 0.0f;
    
    for (uint cl = visibleClustersStartCount.x; cl < visibleClustersStartCount.x + visibleClustersStartCount.y; cl++)
    {
        uint clusterIdx = gVisibleClustersBuffer[cl];
        ClusterData cData = gClusterDataBuffer[clusterIdx];
        
        for (uint vIdx = cData.FirstDataIndex; vIdx < cData.FirstDataIndex + cData.VoxelCount; vIdx++)
        {
            uint voxelIdx = gVoxelsInCluster[vIdx];
            
            if (voxelIdx == faceData.x)
                continue;
            
            float3 currentVoxelWorldCoords = mul(float4(GetVoxelPosition(gVoxelHashedCompactBuffer[voxelIdx], cbVoxelCommons.voxelTextureDimensions), 1.0f), cbVoxelCommons.VoxelToWorld).xyz;
            
            float dist = distance(voxelWorldCoords, currentVoxelWorldCoords);
            
            float valueToAdd = 1.0f - (dist / 5.0f);
            valueToAdd = clamp(valueToAdd, 0.0f, 0.5f);
            
            penaltyValue += valueToAdd;
        }
    }
    
    return sqrt(penaltyValue);
}


[numthreads(128, 1, 1)]
void CS( uint3 DTid : SV_DispatchThreadID )
{
    if (DTid.x >= cbLightTransport.VoxelCount)
        return;
    
    uint2 faceData = gVoxelFaceDataBuffer[DTid.x];
    
    gFaceClusterPenaltyBuffer[DTid.x] = 1.0f;//computeNeighbourIrradiancePenalty(DTid.x, faceData);
    gFaceCloseVoxelsPenaltyBuffer[DTid.x] = 1.0f;//computeCloserVoxelVisibilityFacePenalty(DTid.x, faceData);
}