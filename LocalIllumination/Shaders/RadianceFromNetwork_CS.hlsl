#include "../../VoxelUtils/Shaders/VoxelUtils.hlsli"

ConstantBuffer<ConstantBufferVoxelCommons> cbVoxelCommons : register(b0);

ConstantBuffer<ConstantBufferRadianceFromNetwork> cbRadianceNetwork : register(b1);

StructuredBuffer<uint> gFaceIndexBuffer : register(t0, space0);
StructuredBuffer<uint> gRadianceForFaceBuffer : register(t1, space0);

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
 
    uint faceIdx = gFaceIndexBuffer[DTid.x];
    uint voxIdx = (uint) floor(faceIdx / 6.0f);
    
    gFinalRadianceBuffer[faceIdx] = gRadianceForFaceBuffer[DTid.x];
    SetVoxelPresence(voxIdx, gIndirectLightUpdatedVoxelsBitmap);

}