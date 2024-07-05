#include "VoxelUtils.hlsli"

struct PSInput
{
    float4 position : SV_POSITION;
    float3 normal : NORMAL;
    float3 color : COLOR;
};

cbuffer cbVoxelCommons : register(b0)
{
    VoxelCommons voxelCommons;
}

cbuffer cbCamera : register(b1)
{
    Camera camera;
}

RWTexture3D<float4> gVoxelData : register(u0);

[maxvertexcount(36)]
void GS(
	point float4 input[1] : SV_POSITION, 
	inout TriangleStream< PSInput > output
)
{
    if (!any(gVoxelData[input[0].xyz].xyz))
    {
        return;
    }
    
    
    float3 cubeVertices[8] =
    {
        float3(-1.0, -1.0, 1.0),   // 0
        float3(-1.0, 1.0, 1.0),    // 1
        float3(1.0, 1.0, 1.0),     // 2
        float3(1.0, -1.0, 1.0),    // 3
        float3(-1.0, -1.0, -1.0),    // 4
        float3(-1.0, 1.0, -1.0),     // 5
        float3(1.0, 1.0, -1.0),      // 6
        float3(1.0, -1.0, -1.0)      // 7
    };

    const int numberOfIndices = 36;
    
    int cubeIndices[numberOfIndices] =
    {
    // Front face
    0, 1, 2,
    0, 2, 3,
    
    // Back face
    4, 6, 5,
    4, 7, 6,
    
    // Left face
    4, 5, 1,
    4, 1, 0,
    
    // Right face
    3, 2, 6,
    3, 6, 7,
    
    // Top face
    1, 5, 6,
    1, 6, 2,
    
    // Bottom face
    3, 7, 4,
    3, 4, 0
    };

    float scale = 0.25; // Scale of the cube
    
    
    float3 position = input[0].xyz;
    position.y = (voxelCommons.gridDimension.y - 1) - position.y;
    
    float3 voxelCenter = position * (2.f * scale);
    
    for (int i = 0; i < numberOfIndices; i += 3)
    {
        PSInput output1, output2, output3;

        float3 v1 = scale * cubeVertices[cubeIndices[i]] + voxelCenter;
        float3 v2 = scale * cubeVertices[cubeIndices[i + 1]] + voxelCenter;
        float3 v3 = scale * cubeVertices[cubeIndices[i + 2]] + voxelCenter;

        
        output1.position = mul(float4(v1, 1.0f), camera.ViewProj);
        output2.position = mul(float4(v2, 1.0f), camera.ViewProj);
        output3.position = mul(float4(v3, 1.0f), camera.ViewProj);

        // Calculate face normal
        float3 normal = normalize(cross(v2 - v1, v3 - v1));
        output1.normal = normal;
        output2.normal = normal;
        output3.normal = normal;

        output1.color = gVoxelData[input[0].xyz].xyz;
        output2.color = gVoxelData[input[0].xyz].xyz;
        output3.color = gVoxelData[input[0].xyz].xyz;
        
        output.Append(output1);
        output.Append(output2);
        output.Append(output3);
    }
}