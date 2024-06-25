#include "VoxelUtils.hlsli"


[maxvertexcount(3)]
void GS(
	triangle VertexOut input[3] : SV_POSITION, 
	inout TriangleStream< VertexOut > output
)
{
	// Compute face normal
    float3 faceNormal = abs(normalize(cross(input[1].PosW - input[0].PosW, input[2].PosW - input[0].PosW)));
    float maxNormal = max(faceNormal.x, max(faceNormal.y, faceNormal.z));
	
    float4x4 viewMatrix = vXaxisView;
    uint projAxis = 0;
	
	if (maxNormal == faceNormal.y)
    {
        viewMatrix = vYaxisView;
        projAxis = 1;
    }
	else if (maxNormal == faceNormal.z)
    {
        viewMatrix = vZaxisView;
		projAxis = 2;
    }

	
	for (uint i = 0; i < 3; i++)
	{
		VertexOut element;
		element = input[i];
        element.PosH = mul(element.PosH, mul(viewMatrix, vOrthoProj));
		element.ProjAxis = projAxis;
		output.Append(element);
	}
	
}