#include "VoxelUtils.hlsli"

float4 PS(GeometryOutClusterIndex psIn) : SV_TARGET
{
	return float4(psIn.color, psIn.ClusterIndex);
}