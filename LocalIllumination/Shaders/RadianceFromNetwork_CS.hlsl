#include "../../VoxelUtils/Shaders/VoxelUtils.hlsli"

ConstantBuffer<ConstantBufferVoxelCommons> cbVoxelCommons : register(b0);

ConstantBuffer<ConstantBufferRadianceFromNetwork> cbRadianceNetwork : register(b1);

StructuredBuffer<uint2> gRadianceForFaceBuffer : register(t0, space0);

// UAV 0 not needed
// UAV 1 not needed
// UAV 2 not needed
RWByteAddressBuffer gIndirectLightUpdatedVoxelsBitmap : register(u3);
RWByteAddressBuffer gGaussianUpdatedVoxelsBitmap : register(u4);


RWStructuredBuffer<uint> gFinalRadianceBuffer : register(u0, space1);

[numthreads(128, 1, 1)]
void CS( uint3 DTid : SV_DispatchThreadID )
{
 
    if (DTid.x >= cbRadianceNetwork.ReceivedFaceCount)
        return;
 
    uint2 idxRdx = gRadianceForFaceBuffer[DTid.x];
    
    uint faceIdx = idxRdx.x;
    uint voxIdx = (uint) floor(faceIdx / 6.0f);
    
    gFinalRadianceBuffer[faceIdx] = idxRdx.y;
    SetVoxelPresence(voxIdx, gIndirectLightUpdatedVoxelsBitmap);

}