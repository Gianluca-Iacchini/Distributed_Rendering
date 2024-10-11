#include "VoxelUtils.hlsli"
#define HLSL
#include "TechniquesCompat.h"

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

ConstantBuffer<ConstantBufferVoxelCommons> cbVoxelCommons : register(b0);

cbuffer cbCamera : register(b1)
{
    Camera camera;
}


ByteAddressBuffer gVoxelOccupiedBuffer : register(t0);
StructuredBuffer<FragmentData> gFragmentDataBuffer : register(t1);
StructuredBuffer<uint> gNextIndexBuffer : register(t2);

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space1);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space1);

StructuredBuffer<uint> gVoxelIndicesCompacted : register(t2, space1);
StructuredBuffer<uint> gVoxelHashesCompacted : register(t3, space1);

StructuredBuffer<ClusterData> gClusterDataBuffer : register(t0, space2);
StructuredBuffer<uint> gNextVoxelBuffer : register(t1, space2);
StructuredBuffer<uint> gClusterAssignmentBuffer : register(t2, space2);

StructuredBuffer<uint2> gVoxelFaceDataBuffer : register(t0, space3);

ByteAddressBuffer gVoxelShadowBuffer : register(t0, space4);
ByteAddressBuffer gClusterShadowBuffer : register(t1, space4);
//StructuredBuffer<uint> gVoxelShadowBuffer : register(t0, space4);


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

bool IsVoxelLit(uint voxelIndex, ByteAddressBuffer shadowBuffer)
{
    uint offset = (voxelIndex / 32) * 4;
    uint bitPosition = voxelIndex % 32;
    
    uint shadowVal = shadowBuffer.Load(offset);
    
    return (shadowVal & (1 << bitPosition)) > 0;
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
    
    // Bottom face
    3, 7, 4,
    3, 4, 0,
        
    // Top face
    1, 5, 6,
    1, 6, 2
    };
    
    
    // In the voxel index buffer we have that the index represent the voxel linear coord, while the value represent the index in the
    // fragment buffer of the first fragment present at that voxel coord
    /*
        e.g.
        VoxelIndexBuffer [3, 15, 2, 20, ...,]
        FragmentBuffer [F0, F1, F2, F3, ..., FN]
    
        At index 0 we have the fragment F3. The index 0 represents the voxel linear coord 0, which is (0,0,0) in our world,
    */
    

    
    
    
    uint index = input[0].VoxelIndex;
    uint2 faceData = gVoxelFaceDataBuffer[index];
    
    
    uint voxelLinearCoord = gVoxelHashesCompacted[faceData.x];
    uint fragmentIndex = gVoxelIndicesCompacted[faceData.x];
    

    
    float4 avgColor = float4(0.0f, 0, 0, 1.0f);
    
    uint3 voxelCoord = GetVoxelPosition(voxelLinearCoord, cbVoxelCommons.voxelTextureDimensions);
    int3 iVoxelCoord = int3(voxelCoord);
    
    
    //uint clusterIndex = gClusterAssignmentBuffer[faceData.x];
    //avgColor.xyz = LinearIndexToColor(clusterIndex);
    
    bool isLit = IsVoxelPresent(faceData.x, gVoxelShadowBuffer);
    
    if (isLit)
        avgColor = float4(1.0f, 1.0f, 1.0f, 1.0f);
    else
        avgColor = float4(0.0f, 0.0f, 0.0f, 1.0f);
    
    
    [unroll]
    for (int i = 0; i < 6; i += 3)
    {
        PSInput output;
        
        //float3 v1 = scale * cubeVertices[cubeIndices[((faceData.y * 6) + i)]] + position;
        float3 v1 = VoxelToWorld(voxelCoord + cubeVertices[cubeIndices[((faceData.y * 6) + i)]], cbVoxelCommons);
        //float3 v2 = scale * cubeVertices[cubeIndices[((faceData.y * 6) + i) + 1]] + position;
        float3 v2 = VoxelToWorld(voxelCoord + cubeVertices[cubeIndices[((faceData.y * 6) + i) + 1]], cbVoxelCommons);
        float3 v3 = VoxelToWorld(voxelCoord + cubeVertices[cubeIndices[((faceData.y * 6) + i) + 2]], cbVoxelCommons);
        //float3 v3 = scale * cubeVertices[cubeIndices[((faceData.y * 6) + i) + 2]] + position;
        
        float3 normal = normalize(cross(v2 - v1, v3 - v1));
        output.normal = normal;
        output.color = avgColor.xyz;
        output.ClusterIndex = 1;
        
        output.position = mul(float4(v1, 1.0f), camera.ViewProj);
        triOutput.Append(output);
        
        output.position = mul(float4(v2, 1.0f), camera.ViewProj);
        triOutput.Append(output);
        
        output.position = mul(float4(v3, 1.0f), camera.ViewProj);
        triOutput.Append(output);
        
        triOutput.RestartStrip();
    }
    

}