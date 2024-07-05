#include "VoxelUtils.hlsli"

cbuffer cbVoxelCommons : register(b0)
{
    VoxelCommons voxelCommons;
}

cbuffer cbPerObject : register(b1)
{
    Object object;
}

Texture2D gEmissiveTex : register(t0);
Texture2D gNormalMap : register(t1);
Texture2D gDiffuseTex : register(t2);
Texture2D gMetallicRoughness : register(t3);
Texture2D gOcclusion : register(t4);


StructuredBuffer<float3> gVertices : register(t0, space1);
StructuredBuffer<uint> gIndices : register(t1, space1);
StructuredBuffer<GenericMaterial> gMaterials : register(t2, space1);

RWTexture3D<float4> gVoxelGrid : register(u0);

struct Triangle
{
    float3 v0;
    float3 v1;
    float3 v2;
};

struct Box
{
    float3 center;
    float3 extents;
};

bool OverlapOnAxis(Triangle tri, Box box, float3 axis)
{
    float3 triProjection = float3(
        dot(tri.v0, axis),
        dot(tri.v1, axis),
        dot(tri.v2, axis)
    );
    
    float triMin = min(min(triProjection.x, triProjection.y), triProjection.z);
    float triMax = max(max(triProjection.x, triProjection.y), triProjection.z);
    
    float radius = box.extents.x * axis.x + box.extents.y * axis.y + box.extents.z * axis.z;
    
    return triMin > radius || triMax < -radius;     // if true, no overlap
}

bool TriangleBoxOverlap(Triangle tri, Box box)
{
    float3 edges[3] = { tri.v1 - tri.v0, tri.v2 - tri.v1, tri.v0 - tri.v2 };
    float3 boxAxes[3] = { float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1) };
    
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            float3 axis = cross(edges[i], boxAxes[j]);
            if (OverlapOnAxis(tri, box, axis))
            {
                return false;
            }
        }
    }
    
    return true;
}

[numthreads(8, 8, 8)]
void CS( uint3 DTid : SV_DispatchThreadID )
{
    uint numStructs = 0;
    uint structStride = 0;
    
    gIndices.GetDimensions(numStructs, structStride);
    
    float3 voxelPosition = float3(DTid) * voxelCommons.cellSize;
    
    for (int i = 0; i < numStructs; i+=3)
    {        
        uint3 indices = uint3(gIndices[i], gIndices[i + 1], gIndices[i + 2]);
        float3 v0 = mul(float4(gVertices[indices.x], 1.0f), object.World).xyz - voxelPosition;
        float3 v1 = mul(float4(gVertices[indices.y], 1.0f), object.World).xyz - voxelPosition;
        float3 v2 = mul(float4(gVertices[indices.z], 1.0f), object.World).xyz - voxelPosition;
        
        Triangle tri = { v0, v1, v2 };
        Box box = { voxelPosition, voxelCommons.cellSize };
        
        TriangleBoxOverlap(tri, box);
    }

}