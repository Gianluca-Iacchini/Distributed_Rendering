#include "../../VoxelUtils/Shaders/VoxelUtils.hlsli"

ConstantBuffer<ConstantBufferVoxelCommons> cbCommons : register(b0);
ConstantBuffer<Light> cbLight : register(b1);

Texture2D gShadowMapTexture : register(t0);

ByteAddressBuffer gVoxelOccupiedBuffer : register(t0, space1);

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space2);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space2);
StructuredBuffer<uint> gVoxelIndicesCompactBuffer : register(t2, space2);
StructuredBuffer<uint> gVoxelHashedCompactBuffer : register(t3, space2);

StructuredBuffer<ClusterData> gClusterDataBuffer : register(t0, space3);
StructuredBuffer<uint> gNextVoxelLinkedList : register(t1, space3);
StructuredBuffer<uint> gVoxelAssignmentMap : register(t2, space3);
StructuredBuffer<float3> gVoxelColorBuffer : register(t3, space3);


RWByteAddressBuffer gVoxelLitBuffer : register(u0, space0);
RWStructuredBuffer<uint4> gClusterLitBuffer : register(u1, space0);


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




[numthreads(128, 1, 1)]
void CS( uint3 DTid : SV_DispatchThreadID )
{
    if (DTid.x >= cbCommons.VoxelCount)
        return;
    
    bool lit = false;

    uint linearPos = gVoxelHashedCompactBuffer[DTid.x];
    
    uint3 voxelCoords = GetVoxelPosition(linearPos, cbCommons.voxelTextureDimensions);
    
    float3 worldPos = mul(float4(voxelCoords, 1.0f), cbCommons.VoxelToWorld).xyz;
		
    float3 shadowTestPoints[12];
    
    float3 offset = cbCommons.voxelCellSize;
    offset *= 0.5;
    
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
        shadowCoord = mul(float4(shadowTestPoints[i], 1.0f), cbLight.shadowMatrix);
        shadowCoord /= shadowCoord.w;
        depth = gShadowMapTexture.SampleCmpLevelZero(gShadowSampler, shadowCoord.xy, shadowCoord.z).r;
        if (depth > 0.0f)
        {
            lit = true;
            break;
        }   
    }

    // Test the mid edge points

    // Add some extra distance for edge mid points for voxels in the corner of
	// a rectangular angle, so at least one point comes vsible for the test
    offset *= 1.5; 

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
		
    if (lit == false)
    {
        for (i = 0; i < 12; ++i)
        {
            shadowCoord = mul(float4(shadowTestPoints[i], 1.0f), cbLight.shadowMatrix);
            shadowCoord /= shadowCoord.w;
            depth = gShadowMapTexture.SampleCmpLevelZero(gShadowSampler, shadowCoord.xy, shadowCoord.z).r;
        
            if (depth > 0.0f)
            {
                lit = true;
                break;
            }
        }
    }

    if (lit)
    {
        uint voxelIdx = FindHashedCompactedPositionIndex(voxelCoords, cbCommons.voxelTextureDimensions).x;
        
        bool wasAlreadyLit = SetVoxelPresence(voxelIdx, gVoxelLitBuffer);
        
        if (!wasAlreadyLit)
        {
            uint clusterIdx = gVoxelAssignmentMap[voxelIdx];
            
            if (clusterIdx != UINT_MAX)
            {
                ClusterData clusterData = gClusterDataBuffer[clusterIdx];
        
                float formFactor = differentialAreaFormFactor(clusterData.Normal, cbLight.direction);

                float3 voxelRadiance = formFactor * clusterData.Color;
                uint3 irradianceUint = uint3(voxelRadiance * IRRADIANCE_FIELD_MULTIPLIER);
                
                InterlockedAdd(gClusterLitBuffer[clusterIdx].x, irradianceUint.x);
                InterlockedAdd(gClusterLitBuffer[clusterIdx].y, irradianceUint.y);
                InterlockedAdd(gClusterLitBuffer[clusterIdx].z, irradianceUint.z);
                InterlockedAdd(gClusterLitBuffer[clusterIdx].w, 1);
            }
        }
    }

}