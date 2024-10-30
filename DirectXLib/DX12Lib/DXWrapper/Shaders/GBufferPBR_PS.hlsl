#define PBR 1
#include "Common.hlsli"

struct PSOut
{
    float4 GBufferA : SV_Target0;
    float2 GBufferB : SV_Target1;
    float4 GBufferC : SV_Target2;
    float4 GBufferD : SV_Target3;
    float4 GBufferE : SV_Target4;
};

cbuffer cbPerObject : register(b2)
{
    Object object;
}

cbuffer cbVoxelCommons : register(b3)
{
    uint3 voxelTextureDimensions;
    float totalTime;

    float3 voxelCellSize;
    float deltaTime;

    float3 invVoxelTextureDimensions;
    uint StoreData;

    float3 invVoxelCellSize;
    float pad1;

    float3 SceneAABBMin;
    float pad2;

    float3 SceneAABBMax;
    float pad3;

    float4x4 VoxelToWorld;
    float4x4 WorldToVoxel;
}

Texture2D gEmissiveTex : register(t0);
Texture2D gNormalMap : register(t1);
Texture2D gDiffuseTex : register(t2);
Texture2D gMetallicRoughness : register(t3);
Texture2D gOcclusion : register(t4);

ByteAddressBuffer gVoxelOccupiedBuffer : register(t0, space2);

StructuredBuffer<uint> gIndirectionRankBuffer : register(t0, space3);
StructuredBuffer<uint> gIndirectionIndexBuffer : register(t1, space3);

StructuredBuffer<uint> gVoxelIndicesCompacted : register(t2, space3);
StructuredBuffer<uint> gVoxelHashesCompacted : register(t3, space3);

//StructuredBuffer<ClusterData> gClusterDataBuffer : register(t0, space4);
//StructuredBuffer<uint> gVoxelInClusterBuffer : register(t1, space4);
//StructuredBuffer<uint> gVoxelAssignmentMap : register(t2, space4);
StructuredBuffer<float3> gVoxelColorBuffer : register(t3, space4);

StructuredBuffer<uint2> gPackedRadiance : register(t0, space5);

int3 arrayDirectionTexture[26] =
{
    int3(-1, -1, -1),
    int3(-1, -1, 0),
    int3(-1, -1, 1),
    int3(-1, 0, -1),
    int3(-1, 0, 0),
    int3(-1, 0, 1),
    int3(-1, 1, -1),
    int3(-1, 1, 0),
    int3(-1, 1, 1),
    int3(0, -1, -1),
    int3(0, -1, 0),
    int3(0, -1, 1),
    int3(0, 0, -1),
    int3(0, 0, 1),
    int3(0, 1, -1),
    int3(0, 1, 0),
    int3(0, 1, 1),
    int3(1, -1, -1),
    int3(1, -1, 0),
    int3(1, -1, 1),
    int3(1, 0, -1),
    int3(1, 0, 0),
    int3(1, 0, 1),
    int3(1, 1, -1),
    int3(1, 1, 0),
    int3(1, 1, 1)
};

uint3 worldToVoxelSpace(float3 worldCoordinates)
{
    float3 result = worldCoordinates;
    result -= SceneAABBMin;
    result /= (SceneAABBMax - SceneAABBMin);
    result *= voxelTextureDimensions;
 
    return uint3(uint(result.x), uint(result.y), uint(result.z));
}

float distanceSq(float3 a, float3 b)
{
    float3 d = a - b;
    return dot(d, d);
}

bool IsVoxelPresent(uint voxelLinearCoord)
{
    uint index = voxelLinearCoord >> 5u;
    uint bit = voxelLinearCoord & 31u;
    
    // ByteAddressBuffer operations wants multiple of 4 bytes
    uint value = gVoxelOccupiedBuffer.Load(index * 4);
    
    return (value & (1u << bit)) != 0;
}

uint2 FindHashedCompactedPositionIndex(uint3 coord)
{
    uint2 result = uint2(0, 0); // y field is control value, 0 means element not found, 1 means element found
    uint indirectionIndex = voxelTextureDimensions.z * coord.z + coord.y;
    uint index = gIndirectionIndexBuffer[indirectionIndex];
    uint rank = gIndirectionRankBuffer[indirectionIndex];
    
    uint hashedPosition = coord.x + coord.y * voxelTextureDimensions.x + coord.z * voxelTextureDimensions.x * voxelTextureDimensions.y;
    
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

bool FindClosestOccupiedVoxel(float3 fragmentWS, uint3 voxelCoord, out uint3 outputCoord)
{
    float minimumDistance = 100000000.0f;
    bool foundCoordinates = false;


    uint hashedIndex;
    float3 tempVoxelWorldCoords;
    float distanceSqTemp;

    int3 iVoxelCoord = int3(voxelCoord);
    int3 neighbourCoord;
    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            for (int k = -1; k <= 1; k++)
            {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                
                neighbourCoord = iVoxelCoord + int3(i, j, k);
        
                if (any(neighbourCoord < 0) || any(neighbourCoord >= int3(voxelTextureDimensions)))
                    continue;
        
                hashedIndex = uint(neighbourCoord.x) + uint(neighbourCoord.y) * voxelTextureDimensions.x + uint(neighbourCoord.z) * voxelTextureDimensions.x * voxelTextureDimensions.y;

                if (IsVoxelPresent(uint(hashedIndex)))
                {
                    tempVoxelWorldCoords = mul(float4(float3(neighbourCoord), 1.0f), VoxelToWorld).xyz;
                    distanceSqTemp = distanceSq(fragmentWS, tempVoxelWorldCoords);

                    if (distanceSqTemp < minimumDistance)
                    {
                        minimumDistance = distanceSqTemp;
                        outputCoord = uint3(neighbourCoord);
                        foundCoordinates = true;
                    }
                }
            }
        }

    }

    return foundCoordinates;
}

int FindMostAlignedDirection(float3 N)
{
    float3 faceDir[6] =
    {
        float3(0.0f, 0.0f, -1.0f),
        float3(0.0f, 0.0f, 1.0f),
        float3(-1.0f, 0.0f, 0.0f),
        float3(1.0f, 0.0f, 0.0f),
        float3(0.0f, -1.0f, 0.0f),
        float3(0.0f, 1.0f, 0.0f)
    };
    
    int bestIndex = 0;
    float maxDot = dot(N, faceDir[0]); // Initialize with the first direction

    // Loop through the rest of the face directions
    for (int i = 1; i < 6; i++)
    {
        float currentDot = dot(N, faceDir[i]);
        if (currentDot > maxDot)
        {
            maxDot = currentDot;
            bestIndex = i;
        }
    }

    return bestIndex;
}


PSOut PS(VertexOutPosNormalTex pIn)
{
    float4 diffuse = gDiffuseTex.Sample(gSampler, pIn.Tex);
    
#ifdef ALPHA_TEST
    if (diffuse.a < 0.1f)
    {
        discard;
    }
#endif
    
    PSOut psOut;
    
    pIn.NormalW = normalize(pIn.NormalW);
    
    // Get normal from normal map. Compute Z from X and Y.
    float3 normalMapSample = ComputeTwoChannelNormal(gNormalMap.Sample(gSampler, pIn.Tex).xy);
    // Compute TBN matrix from position, normal and texture coordinates.
    float3x3 tbn = CalculateTBN(pIn.PosW, pIn.NormalW, pIn.Tex);
    // Transform normal from tangent space to world space
    float3 normal = normalize(mul(normalMapSample, tbn));
    
    psOut.GBufferA = float4(pIn.PosW, object.MaterialIndex);
    psOut.GBufferB = PackNormal(normal);
    psOut.GBufferC = diffuse;
    
    // w coordinated is used to keep track of geometry. Pixels where there is no geometry have a w value of 1.0f
    // due to the fact that the clear color used for this GBuffer is red (1.0f, 0.0f, 0.0f, 1.0f)
    psOut.GBufferD = float4(gMetallicRoughness.Sample(gSampler, pIn.Tex).rgb, 0.0f);
    
    float3 worldCoord = pIn.PosW;
    uint3 voxelCoord = uint3(mul(float4(worldCoord, 1.0f), WorldToVoxel).xyz);
    uint linearCoord = voxelCoord.x + voxelCoord.y * voxelTextureDimensions.x + voxelCoord.z * voxelTextureDimensions.x * voxelTextureDimensions.y;
    
    uint faceDir = FindMostAlignedDirection(normal);
    
    float4 radiance = float4(0.0f, 0.0f, 0.0f, 0.0f);
    bool found = IsVoxelPresent(linearCoord);
    if (!found)
    {
        uint3 newCoords;
        found = FindClosestOccupiedVoxel(worldCoord, voxelCoord, newCoords);
        voxelCoord = newCoords;
    }
    if (found)
    {
        radiance = float4(1.0f, 1.0f, 1.0f, 1.0f);
        linearCoord = voxelCoord.x + voxelCoord.y * voxelTextureDimensions.x + voxelCoord.z * voxelTextureDimensions.x * voxelTextureDimensions.y;
        uint2 result = FindHashedCompactedPositionIndex(voxelCoord);
        uint2 packedRadiance = gPackedRadiance[result.x * 6 + faceDir];
        radiance.xy = UnpackFloats16(packedRadiance.x);
        radiance.zw = UnpackFloats16(packedRadiance.y);
    }
    
    psOut.GBufferE = radiance;
    
    return psOut;
}