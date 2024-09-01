#include "VoxelUtils.hlsli"

struct PSInput
{
    float4 position : SV_POSITION;
    float3 normal : NORMAL;
    float3 color : COLOR;
    uint ClusterIndex : CLUSTERINDEX;
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

StructuredBuffer<FragmentData> gFragmentDataBuffer : register(t0);
StructuredBuffer<uint> gNextIndexBuffer : register(t1);

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space1);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space1);

StructuredBuffer<uint> gVoxelIndicesCompacted : register(t2, space1);
StructuredBuffer<uint> gVoxelHashesCompacted : register(t3, space1);

StructuredBuffer<ClusterData> gClusterDataBuffer : register(t0, space2);
StructuredBuffer<uint> gNextVoxelBuffer : register(t1, space2);
StructuredBuffer<uint> gClusterAssignmentBuffer : register(t2, space2);

RWStructuredBuffer<float3> gVoxelNormalBuffer : register(u7, space0);

uint3 GetVoxelPosition(uint voxelLinearCoord)
{
    uint3 voxelPosition;
    voxelPosition.x = voxelLinearCoord % voxelCommons.gridDimension.x;
    voxelPosition.y = (voxelLinearCoord / voxelCommons.gridDimension.x) % voxelCommons.gridDimension.y;
    voxelPosition.z = voxelLinearCoord / (voxelCommons.gridDimension.x * voxelCommons.gridDimension.y);
    return voxelPosition;
}


uint2 FindHashedCompactedPositionIndex(uint3 coord, uint3 gridDimension)
{
    uint2 result = uint2(0, 0); // y field is control value, 0 means element not found, 1 means element found
    uint indirectionIndex = gridDimension.z * coord.z + coord.y;
    uint index = gIndirectionIndexBuffer[indirectionIndex];
    uint rank = gIndirectionRankBuffer[indirectionIndex];
    uint hashedPosition = GetLinearCoord(coord, gridDimension);
    
    if (rank == 0)
        return result;
    
    uint tempHashed;
    uint startIndex = index;
    uint endIndex = index + rank;
    uint currentIndex = (startIndex + endIndex) / 2;

    for (int i = 0; i < int(12); ++i)
    {
        tempHashed = gVoxelHashesCompacted[currentIndex];

        if (tempHashed == hashedPosition)
        {
            return uint2(currentIndex, 1);
        }

        if (tempHashed < hashedPosition)
        {
            startIndex = currentIndex;
            currentIndex = (startIndex + endIndex) / 2;
        }
        else
        {
            endIndex = currentIndex;
            currentIndex = (startIndex + endIndex) / 2;
        }
    }

    return result;
}




float3 LinearIndexToColor(uint index)
{
    if (index == UINT_MAX)
        return float3(0, 0, 0);
    else if (index == 0)
        return float3(1, 1, 1);
    
    // Hash the index to produce a pseudo-random float3 color
        uint hash = index;

    // Example hash function (based on bitwise operations)
    hash = (hash ^ 61) ^ (hash >> 16);
    hash = hash + (hash << 3);
    hash = hash ^ (hash >> 4);
    hash = hash * 0x27d4eb2d;
    hash = hash ^ (hash >> 15);

    // Convert the hash to a float3 color in the range [0, 1]
    float r = (float) ((hash >> 16) & 0xFF) / 255.0;
    float g = (float) ((hash >> 8) & 0xFF) / 255.0;
    float b = (float) (hash & 0xFF) / 255.0;

    if (r < 0.1 && g < 0.1 && b < 0.1)
    {
        r = 0.39f;
        g = 0.39f;
        b = 0.39f;
    }
    
        return float3(r, g, b);
}

[maxvertexcount(72)]
void GS(
	point GSInput input[1], 
	inout TriangleStream< PSInput > triOutput
)
{

    float3 cubeVertices[8] =
    {
        float3(-0.5f, -0.5f, -0.5f), // 0
        float3(-0.5f, 0.5f, -0.5f), // 1
        float3(0.5f, 0.5f, -0.5f), // 2
        float3(0.5f, -0.5f, -0.5f), // 3
        float3(-0.5f, -0.5f, 0.5f), // 4
        float3(-0.5f, 0.5f, 0.5f), // 5
        float3(0.5f, 0.5f, 0.5f), // 6
        float3(0.5f, -0.5f, 0.5f) // 7
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
    
    float4 avgColor = float4(0, 0, 0, 0);
    
    uint index = input[0].VoxelIndex;
    
    uint voxelLinearCoord = gVoxelHashesCompacted[index];
    uint fragmentIndex = gVoxelIndicesCompacted[index];
    
    uint fragmentCount = 0;
    
    
    //while (fragmentIndex != UINT_MAX)
    //{
    //    avgColor += gFragmentDataBuffer[fragmentIndex].color;
    //    fragmentIndex = gNextIndexBuffer[fragmentIndex];
    //    fragmentCount += 1;
    //}
   
    //avgColor = avgColor / fragmentCount;
    avgColor.xyz = float3(0.0f, 0.0f, 0.0f);
    uint clusterIndex = gClusterAssignmentBuffer[index];
    avgColor.xyz = LinearIndexToColor(clusterIndex);
    
    //for (uint j = 0; j < 210; j++)
    //{
    //    if (gClusterAssignmentBuffer[j + 100000] == gClusterAssignmentBuffer[index])
    //        avgColor.xyz = float3(j * 0.1f, 1.0f - j * 0.1f, 0.0f);

    //}
    

    //avgColor.xyz = gVoxelNormalBuffer[index];
    //avgColor.xyz = avgColor.xyz;
    
    //avgColor.x = avgColor.x == -1 ? 0.15f : avgColor.x;
    //avgColor.y = avgColor.y == -1 ? 0.15f : avgColor.y;
    //avgColor.z = avgColor.z == -1 ? 0.15f : avgColor.z;
    
    float scale = 0.5f; // Scale of the cube
    
    
    uint3 voxelPosition = GetVoxelPosition(voxelLinearCoord);
    
    float3 position = float3(voxelPosition);
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
        output.ClusterIndex = clusterIndex;
        
        output.position = mul(float4(v1, 1.0f), camera.ViewProj);
        triOutput.Append(output);
        
        output.position = mul(float4(v2, 1.0f), camera.ViewProj);
        triOutput.Append(output);
        
        output.position = mul(float4(v3, 1.0f), camera.ViewProj);
        triOutput.Append(output);

        triOutput.RestartStrip();
    }
    

}