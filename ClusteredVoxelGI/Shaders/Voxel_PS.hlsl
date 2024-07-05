#include "VoxelUtils.hlsli"

Texture2D gEmissiveTex : register(t0);
Texture2D gNormalMap : register(t1);
Texture2D gDiffuseTex : register(t2);
Texture2D gMetallicRoughness : register(t3);
Texture2D gOcclusion : register(t4);

RWTexture3D<float4> gVoxelGrid : register(u0);

cbuffer cbVoxelCommons : register(b0)
{
    VoxelCommons voxelCommons;
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


    gVoxelGrid[voxelTexCoord] = max(gVoxelGrid[voxelTexCoord], diffuse);
   
    
    
    return retColor;

}