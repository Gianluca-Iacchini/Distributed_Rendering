#include "VoxelUtils.hlsli"



Texture2D gEmissiveTex : register(t0);
Texture2D gNormalMap : register(t1);
Texture2D gDiffuseTex : register(t2);
Texture2D gMetallicRoughness : register(t3);
Texture2D gOcclusion : register(t4);


RWStructuredBuffer<uint> gFragmentCounter : register(u0);
RWStructuredBuffer<uint> gOccupiedVoxelCounter : register(u1);
RWStructuredBuffer<uint> gVoxelOccupiedBuffer : register(u2);
RWStructuredBuffer<uint> gVoxelIndicesBuffer : register(u3);

RWStructuredBuffer<FragmentData> gFragmentDataBuffer : register(u4);



RWStructuredBuffer<uint> gNextIndexBuffer : register(u5);

RWStructuredBuffer<uint> gVoxelHashBuffer : register(u6);




cbuffer cbVoxelCommons : register(b0)
{
    VoxelCommons voxelCommons;
}

bool isVoxelOccupied(uint value, uint bit)
{
    bool result = ((value & (1 << bit)) > 0);
    return result;
}

void SetVoxelOccupied(uint indexHashed)
{
    float integerPart;
    float indexDecimalFloat = float(indexHashed) / 32.0;
    float fractional = modf(indexDecimalFloat, integerPart);
    uint index = uint(integerPart);
    uint bit = uint(fractional * 32.0);
    uint value = (1 << bit);
    uint originalValue = 0;
    InterlockedOr(gVoxelOccupiedBuffer[index], value, originalValue);

    if (isVoxelOccupied(originalValue, bit) == false)
    {
        InterlockedAdd(gOccupiedVoxelCounter[0], 1);
    }
}


float4 PS(VertexOutVoxel pIn) : SV_TARGET
{
    float4 diffuse = gDiffuseTex.Sample(gSampler, pIn.Tex);
    
    float4 retColor = float4(1.0f, 1.0f, 1.0f, 1.0f);
    uint3 voxelTexCoord = uint3(0, 0, 0);
    
    // Projecting along X axis
    if (pIn.ProjAxis == 0)
    {
        voxelTexCoord.x = min(voxelCommons.gridDimension.x * pIn.PosH.z, (voxelCommons.gridDimension.x - 1));
        voxelTexCoord.y = pIn.PosH.y;
        voxelTexCoord.z = (voxelCommons.gridDimension.z - 1) - pIn.PosH.x;
        retColor = float4(1.0f, 0.0f, 0.0f, 1.0f);
    }
    // Projection along Y axis
    else if (pIn.ProjAxis == 1)
    {
        voxelTexCoord.x = (voxelCommons.gridDimension.x - 1) - pIn.PosH.x;
        voxelTexCoord.y = (voxelCommons.gridDimension.y - 1) - min(voxelCommons.gridDimension.y * pIn.PosH.z, (voxelCommons.gridDimension.y - 1));
        voxelTexCoord.z = (voxelCommons.gridDimension.z - 1) - pIn.PosH.y;
        retColor = float4(0.0f, 1.0f, 0.0f, 1.0f);
    }
    else if (pIn.ProjAxis == 2)
    {
        voxelTexCoord.x = pIn.PosH.x;
        voxelTexCoord.y = pIn.PosH.y;
        voxelTexCoord.z = min(voxelCommons.gridDimension.z * pIn.PosH.z, (voxelCommons.gridDimension.z - 1));
        retColor = float4(0.0f, 0.0f, 1.0f, 1.0f);
    }
    

    
    uint lastFragmentCount = 0;
    
    InterlockedAdd(gFragmentCounter[0], 1, lastFragmentCount);
    

    
    if (voxelCommons.storeData > 0)
    {
        uint voxelLinearCoord =
            voxelTexCoord.x +
            voxelTexCoord.y * voxelCommons.gridDimension.x +
            voxelTexCoord.z * voxelCommons.gridDimension.x * voxelCommons.gridDimension.y;
        
        FragmentData fragmentData;
        fragmentData.color = diffuse;
        fragmentData.normal = float3(0, 0, 0);
        fragmentData.position = pIn.PosW;
        fragmentData.voxelLinearCoord = voxelLinearCoord;
        fragmentData.pad0 = 0.0f;
        
        gFragmentDataBuffer[lastFragmentCount] = fragmentData;
        gNextIndexBuffer[lastFragmentCount] = UINT_MAX;
        
        //gVoxelIndicesBuffer[voxelLinearCoord] = lastFragmentCount;
        
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
    
    return retColor;

}