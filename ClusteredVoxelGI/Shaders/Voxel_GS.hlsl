#include "VoxelUtils.hlsli"


[maxvertexcount(3)]
void GS(
	triangle VertexOut input[3] : SV_POSITION, 
	inout TriangleStream< VertexOut > output
)
{
	for (uint i = 0; i < 3; i++)
	{
		VertexOut element;
		element = input[i];
		output.Append(element);
	}
}