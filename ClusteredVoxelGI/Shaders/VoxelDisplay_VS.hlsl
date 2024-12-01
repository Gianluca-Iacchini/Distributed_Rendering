#include "VoxelUtils.hlsli"

VertexOutVoxelIndex VS(uint coord : SV_Position)
{
    VertexOutVoxelIndex gsInput;
    gsInput.Pos = float4(coord, 0.0f, 0.0f, 1.0f);
    gsInput.VoxelIndex = coord;
    
    return gsInput;
}