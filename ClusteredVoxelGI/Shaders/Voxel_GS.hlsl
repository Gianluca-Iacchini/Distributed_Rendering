#include "../../VoxelUtils/Shaders/VoxelUtils.hlsli"

cbuffer cbVoxelCamera : register(b1)
{
    VoxelCamera voxelCamera;
}

[maxvertexcount(3)]
void GS(
	triangle VertexOutVoxel input[3], 
	inout TriangleStream< VertexOutVoxel > output
)
{
	// Compute face normal
    float3 faceNormal = abs(normalize(cross(input[1].PosW - input[0].PosW, input[2].PosW - input[0].PosW)));
    float maxNormal = max(faceNormal.x, max(faceNormal.y, faceNormal.z));
	
    float4x4 viewMatrix = voxelCamera.xAxisView;
    uint projAxis = 0;
	
	if (maxNormal == faceNormal.y)
    {
        viewMatrix = voxelCamera.yAxisView;
        projAxis = 1;
    }
	else if (maxNormal == faceNormal.z)
    {
        viewMatrix = voxelCamera.zAxisView;
		projAxis = 2;
    }

	
	for (uint i = 0; i < 3; i++)
	{
		VertexOutVoxel element;
		element = input[i];
        element.PosH = mul(element.PosH, mul(viewMatrix, voxelCamera.orthoProj));
		element.ProjAxis = projAxis;
		element.Tex = input[i].Tex;
		output.Append(element);
	}
	
}