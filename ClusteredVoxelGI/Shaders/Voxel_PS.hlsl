#include "../../VoxelUtils/Shaders/VoxelUtils.hlsli"


Texture2D gEmissiveTex : register(t0);
Texture2D gNormalMap : register(t1);
Texture2D gDiffuseTex : register(t2);
Texture2D gMetallicRoughness : register(t3);
Texture2D gOcclusion : register(t4);


globallycoherent RWByteAddressBuffer gVoxelOccupiedBuffer : register(u0);
globallycoherent RWStructuredBuffer<FragmentData> gFragmentDataBuffer : register(u1);
globallycoherent RWStructuredBuffer<uint> gNextIndexBuffer : register(u2);
globallycoherent RWStructuredBuffer<uint> gVoxelIndicesBuffer : register(u3);
globallycoherent RWStructuredBuffer<uint> gFragmentCounter : register(u4);
globallycoherent RWStructuredBuffer<uint> gOccupiedVoxelCounter : register(u5);
globallycoherent RWStructuredBuffer<uint> gVoxelHashBuffer : register(u6);




cbuffer cbVoxelCommons : register(b0)
{
    VoxelCommons voxelCommons;
}

cbuffer RootConstant : register(b3)
{
    uint StoreData;
}

void SetVoxelOccupied(uint indexHashed)
{
    bool wasAlreadyOccupied = SetVoxelPresence(indexHashed, gVoxelOccupiedBuffer);
    // Check if the bit was previously unset and atomically increment the counter if it was
    if (!wasAlreadyOccupied)
    {
        InterlockedAdd(gOccupiedVoxelCounter[0], 1);
    }
}


void PS(VertexOutVoxel pIn)
{   
    pIn.NormalW = normalize(pIn.NormalW);
    
    // Get normal from normal map. Compute Z from X and Y.
    float3 normalMapSample = ComputeTwoChannelNormal(gNormalMap.Sample(gSampler, pIn.Tex).xy);
    // Compute TBN matrix from position, normal and texture coordinates.
    float3x3 tbn = CalculateTBN(pIn.PosW, pIn.NormalW, pIn.Tex);
    // Transform normal from tangent space to world space
    float3 normal = normalize(mul(normalMapSample, tbn));
    
    float4 diffuse = gDiffuseTex.Sample(gSampler, pIn.Tex);
    
    float4 retColor = float4(1.0f, 1.0f, 1.0f, 1.0f);
    uint3 voxelTexCoord = uint3(0, 0, 0);
    
    // Projecting along X axis
    if (pIn.ProjAxis == 0)
    {
        voxelTexCoord.x = min(voxelCommons.gridDimension.x * pIn.PosH.z, (voxelCommons.gridDimension.x - 1));
        voxelTexCoord.y = (voxelCommons.gridDimension.y - 1) - pIn.PosH.y;
        voxelTexCoord.z = (voxelCommons.gridDimension.z - 1) - pIn.PosH.x;
        retColor = float4(1.0f, 0.0f, 0.0f, 1.0f);
    }
    // Projection along Y axis
    else if (pIn.ProjAxis == 1)
    {
        voxelTexCoord.x = (voxelCommons.gridDimension.x - 1) - pIn.PosH.x;
        voxelTexCoord.y = min(voxelCommons.gridDimension.y * pIn.PosH.z, (voxelCommons.gridDimension.y - 1));
        voxelTexCoord.z = (voxelCommons.gridDimension.z - 1) - pIn.PosH.y;
        retColor = float4(0.0f, 1.0f, 0.0f, 1.0f);
    }
    else if (pIn.ProjAxis == 2)
    {
        voxelTexCoord.x = pIn.PosH.x;
        voxelTexCoord.y = (voxelCommons.gridDimension.y - 1) - pIn.PosH.y;
        voxelTexCoord.z = min(voxelCommons.gridDimension.z * pIn.PosH.z, (voxelCommons.gridDimension.z - 1));
        retColor = float4(0.0f, 0.0f, 1.0f, 1.0f);
    }
    

    
    uint lastFragmentCount = 0;
    
    InterlockedAdd(gFragmentCounter[0], 1, lastFragmentCount);
    

    
    if (StoreData > 0)
    {
        uint voxelLinearCoord =
            voxelTexCoord.x +
            voxelTexCoord.y * voxelCommons.gridDimension.x +
            voxelTexCoord.z * voxelCommons.gridDimension.x * voxelCommons.gridDimension.y;
        
        FragmentData fragmentData;
        fragmentData.color = diffuse;
        fragmentData.normal = normal;
        fragmentData.position = pIn.PosW;
        fragmentData.voxelLinearCoord = voxelLinearCoord;
        fragmentData.pad0 = 0.0f;
        
        gFragmentDataBuffer[lastFragmentCount] = fragmentData;
        gNextIndexBuffer[lastFragmentCount] = UINT_MAX;
        
        
        uint newVal = lastFragmentCount;
        uint prev = UINT_MAX;
        
        uint currentValue;
        InterlockedCompareExchange(gVoxelIndicesBuffer[voxelLinearCoord], prev, newVal, currentValue);
        
        [allow_uav_condition]
        while (currentValue != prev)
        {

            prev = currentValue;
            gNextIndexBuffer[lastFragmentCount] = currentValue;
            InterlockedCompareExchange(gVoxelIndicesBuffer[voxelLinearCoord], prev, newVal, currentValue);
        }
        
        SetVoxelOccupied(voxelLinearCoord);
    }
 
}