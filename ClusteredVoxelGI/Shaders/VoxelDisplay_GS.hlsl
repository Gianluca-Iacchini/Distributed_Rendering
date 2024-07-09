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

Texture3D<uint4> gVoxelData : register(t0);

[maxvertexcount(36)]
void GS(
	point float4 input[1] : SV_POSITION, 
	inout TriangleStream< PSInput > triOutput
)
{

    float3 cubeVertices[8] =
    {
        float3(-0.5f, -0.5f, 0.5f), // 0
        float3(-0.5f, 0.5f, 0.5f), // 1
        float3(0.5f, 0.5f, 0.5f), // 2
        float3(0.5f, -0.5f, 0.5f), // 3
        float3(-0.5f, -0.5f, -0.5f), // 4
        float3(-0.5f, 0.5f, -0.5f), // 5
        float3(0.5f, 0.5f, -0.5f), // 6
        float3(0.5f, -0.5f, -0.5f) // 7
    };
    
    int cubeIndices[36] =
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
    
    
    uint3 voxelColor = gVoxelData[input[0].xyz].xyz;
    
    if (!any(voxelColor))
    {
        return;
    }
    
   

    float scale = 0.5; // Scale of the cube
    
    
    float3 position = input[0].xyz;
    position.y = (voxelCommons.gridDimension.y - 1) - position.y;
    position = position * scale + float3(0.5f, 0.5f, 0.5f);
    
    for (int i = 0; i < 36; i += 3)
    {
        PSInput output;

        float3 v1 = scale * cubeVertices[cubeIndices[i]] + position;
        float3 v2 = scale * cubeVertices[cubeIndices[i + 1]] + position;
        float3 v3 = scale * cubeVertices[cubeIndices[i + 2]] + position;

        float3 normal = normalize(cross(v2 - v1, v3 - v1));
        output.normal = normal;
        output.color = (float3) voxelColor / 255.0f;
        
        output.position = mul(float4(v1, 1.0f), camera.ViewProj);
        triOutput.Append(output);
        
        output.position = mul(float4(v2, 1.0f), camera.ViewProj);
        triOutput.Append(output);
        
        output.position = mul(float4(v3, 1.0f), camera.ViewProj);
        triOutput.Append(output);

    }
}