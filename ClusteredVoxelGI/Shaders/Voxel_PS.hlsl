#include "VoxelUtils.hlsli"

float4 PS(VertexOut pIn) : SV_TARGET
{
    float4 retColor = float4(1.0f, 1.0f, 1.0f, 1.0f);
    uint3 voxelTexCoord = uint3(0, 0, 0);
    
    // Projecting along X axis
    if (pIn.ProjAxis == 0)
    {
        voxelTexCoord.x = min(vVoxelGridDimension.x * pIn.PosH.z, (vVoxelGridDimension.x - 1));
        voxelTexCoord.y = pIn.PosH.y;
        voxelTexCoord.z = (vVoxelGridDimension.z - 1) - pIn.PosH.x;
        retColor = float4(1.0f, 0.0f, 0.0f, 1.0f);
    }
    // Projection along Y axis
    else if (pIn.ProjAxis == 1)
    {
        voxelTexCoord.x = (vVoxelGridDimension.x - 1) - pIn.PosH.x;
        voxelTexCoord.y = (vVoxelGridDimension.y - 1) - min(vVoxelGridDimension.y * pIn.PosH.z, (vVoxelGridDimension.y - 1));
        voxelTexCoord.z = (vVoxelGridDimension.z - 1) - pIn.PosH.y;
        retColor = float4(0.0f, 1.0f, 0.0f, 1.0f);
    }
    else if (pIn.ProjAxis == 2)
    {
        voxelTexCoord.x = pIn.PosH.x;
        voxelTexCoord.y = pIn.PosH.y;
        voxelTexCoord.z = min(vVoxelGridDimension.z * pIn.PosH.z, (vVoxelGridDimension.z - 1));
        retColor = float4(0.0f, 0.0f, 1.0f, 1.0f);
    }

    gVoxelGrid[voxelTexCoord] = retColor;
    
    return retColor;

}