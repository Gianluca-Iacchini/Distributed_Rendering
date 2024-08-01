#include "VoxelUtils.hlsli"

struct PSInput
{
    float4 position : SV_POSITION;
    float3 normal : NORMAL;
    float3 color : COLOR;
};

struct GSInput
{
    float4 Pos : SV_Position;
    uint VoxelIndex : VOXELINDEX;
};

cbuffer cbVoxelCommons : register(b0)
{
    VoxelCommons voxelCommons;
}

cbuffer cbCamera : register(b1)
{
    Camera camera;
}
RWStructuredBuffer<uint> gFragmentCounter : register(u0);
RWStructuredBuffer<uint> gOccupiedVoxelCounter : register(u1);
RWStructuredBuffer<uint> gVoxelOccupiedBuffer : register(u2);
RWStructuredBuffer<uint> gVoxelIndicesBuffer : register(u3);

RWStructuredBuffer<FragmentData> gFragmentDataBuffer : register(u4);

RWStructuredBuffer<uint> gNextIndexBuffer : register(u5);

RWStructuredBuffer<uint> gVoxelHashBuffer : register(u6);




uint3 GetVoxelPosition(uint voxelLinearCoord)
{
    uint3 voxelPosition;
    voxelPosition.x = voxelLinearCoord % voxelCommons.gridDimension.x;
    voxelPosition.y = (voxelLinearCoord / voxelCommons.gridDimension.x) % voxelCommons.gridDimension.y;
    voxelPosition.z = voxelLinearCoord / (voxelCommons.gridDimension.x * voxelCommons.gridDimension.y);
    return voxelPosition;
}

[maxvertexcount(36)]
void GS(
	point GSInput input[1], 
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
    
    // In the voxel index buffer we have that the index represent the voxel linear coord, while the value represent the index in the
    // fragment buffer of the first fragment present at that voxel coord
    /*
        e.g.
        VoxelIndexBuffer [3, 15, 2, 20, ...,]
        FragmentBuffer [F0, F1, F2, F3, ..., FN]
    
        At index 0 we have the fragment F3. The index 0 represents the voxel linear coord 0, which is (0,0,0) in our world,
    */
    uint fragmentIndex = input[0].VoxelIndex;
    
    float4 avgColor = float4(0, 0, 0, 0);
    uint voxelLinearCoord = gFragmentDataBuffer[fragmentIndex].voxelLinearCoord;
    uint fragmentCount = 0;
    
    while (fragmentIndex != UINT_MAX)
    {
        avgColor += gFragmentDataBuffer[fragmentIndex].color;
        fragmentIndex = gNextIndexBuffer[fragmentIndex];
        fragmentCount += 1;
    }
   
    avgColor = avgColor / fragmentCount;
    
    float scale = 0.5f; // Scale of the cube
    
    
    float3 position = GetVoxelPosition(voxelLinearCoord);
    position.y = (voxelCommons.gridDimension.y - 1) - position.y;
    position = position * scale + float3(0.5f, 0.5f, 0.5f);
    
    // Move voxel scene to position 0,0,0
    position -= voxelCommons.gridDimension * (scale / 2.f);
    
    for (int i = 0; i < 36; i += 3)
    {
        PSInput output;

        float3 v1 = scale * cubeVertices[cubeIndices[i]] + position;
        float3 v2 = scale * cubeVertices[cubeIndices[i + 1]] + position;
        float3 v3 = scale * cubeVertices[cubeIndices[i + 2]] + position;

        float3 normal = normalize(cross(v2 - v1, v3 - v1));
        output.normal = normal;
        output.color = avgColor.xyz;
        
        output.position = mul(float4(v1, 1.0f), camera.ViewProj);
        triOutput.Append(output);
        
        output.position = mul(float4(v2, 1.0f), camera.ViewProj);
        triOutput.Append(output);
        
        output.position = mul(float4(v3, 1.0f), camera.ViewProj);
        triOutput.Append(output);

    }
}