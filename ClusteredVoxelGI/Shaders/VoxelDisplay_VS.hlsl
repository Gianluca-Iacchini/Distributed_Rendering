
struct GSInput
{
    float4 Pos : SV_Position;
    uint VoxelIndex : VOXELINDEX;
};

GSInput VS(uint coord : SV_Position)
{
    GSInput gsInput;
    gsInput.Pos = float4(coord, 0.0f, 0.0f, 1.0f);
    gsInput.VoxelIndex = coord;
    
    return gsInput;
}