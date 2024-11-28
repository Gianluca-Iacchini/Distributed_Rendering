#include "Common.hlsli"

struct PSOut
{
    float4 DeferredTexture : SV_Target0;
    float4 RadianceTexture : SV_Target1;
};

cbuffer cbCommons : register(b0)
{
    Commons commons;
};

cbuffer cbCamera : register(b1)
{
    Camera camera;
}

cbuffer cbVoxelCommons : register(b2)
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

Texture2D gShadowMap : register(t0);
Texture2D gCameraDepth : register(t1);

Texture2D gBufferaWorld : register(t2);
Texture2D gBufferNormal : register(t3);
Texture2D gBufferDiffuse : register(t4);
Texture2D gBufferMetallicRoughnessAO : register(t5);

StructuredBuffer<Light> gLights : register(t0, space1);
StructuredBuffer<GenericMaterial> gMaterials : register(t1, space1);

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

static const uint UINT_MAX = 0xFFFFFFFF;

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
    
    if (rank > 0)
    {
        uint tempHashed;
        uint startIndex = index;
        uint endIndex = index + rank;
        uint currentIndex = (startIndex + endIndex) / 2;

        for (int i = 0; i < int(12); ++i)
        {
            tempHashed = gVoxelHashesCompacted[currentIndex];

            if (tempHashed == hashedPosition)
            {
                result = uint2(currentIndex, 1);
                break;
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

float3 pointPlaneProjection(float3 pnt, float3 planeNormal, float3 planePoint)
{
    return pnt - dot(planeNormal, float3(pnt - planePoint)) * planeNormal;
}

float3 pointSegmentProjection(float3 s0, float3 s1, float3 p)
{
    float3 a = p - s0;
    float3 b = s1 - s0;
    float3 aProjectedOverB = (dot(a, b) / dot(b, b)) * b;

    return s0 + aProjectedOverB;
}

float3 pointSegmentProjectionFixEdges(float3 s0, float3 s1, float3 p)
{
    float3 a = p - s0;
    float3 b = s1 - s0;
    float3 aProjectedOverB = (dot(a, b) / dot(b, b)) * b;
    float3 projectedPoint = s0 + aProjectedOverB;
    
    float t = dot(a, b) / dot(b, b);
    
    projectedPoint = (t < 0.0f) ? s0 : (t > 1.0f) ? s1 : projectedPoint;
    
    return projectedPoint;
}

float3 approximateIrradianceTwoPoint(float3 p0, float3 p1, float3 i0, float3 i1, float3 pnt)
{
    float3 projection = pointSegmentProjectionFixEdges(p0, p1, pnt);
    float segmentLength = distance(p0, p1) + 0.001f;
    float p0ProjectionDistance = distance(p0, projection);
    
    float3 finalIrradiance = lerp(i0, i1, (p0ProjectionDistance / segmentLength));
    
    return finalIrradiance;
}

//float3 InterpolateIrradiance(uint3 voxelCoords, float3 fragmentWorldPos, uint index, bool isDisplaced)
//{  
//    float3 voxelWorldCoords = mul(float4(float3(voxelCoords), 1.0f), VoxelToWorld).xyz;
//    float3 voxelCenterToFragment = normalize(fragmentWorldPos - voxelWorldCoords);

//    int3 iVoxelCoords = int3(voxelCoords);
    
//    bool hasBigDisplacement = isDisplaced && (distance(fragmentWorldPos, voxelWorldCoords) > voxelCellSize.x / 2.0f);

    
//    int3 offset0 = int3(sign(voxelCenterToFragment.x), 0, 0);
//    int3 offset1 = int3(0, sign(voxelCenterToFragment.y), 0);
//    int3 offset2 = int3(0, 0, sign(voxelCenterToFragment.z));
    
//    bool isNotAligned = false;
    
//    uint voxelIdx = FindHashedCompactedPositionIndex(voxelCoords).x;

//    uint faceIdx = voxelIdx * 6 + index;
    
    
//    uint2 packedRadiance = gPackedRadiance[faceIdx];
    
//    float3 radiance = float3(0.0f, 0.0f, 0.0f);
    
//    radiance.xy = UnpackFloats16(packedRadiance.x);
//    radiance.z = UnpackFloats16(packedRadiance.y).x;

//    //int3 iNeighbourCoords[7];
    
//    //uint arrIndex = 0;
    
//    //for (int x = 0; x <= 1; x++)
//    //{
//    //    for (int y = 0; y <= 1; y++)
//    //    {
//    //        for (int z = 0; z <= 1; z++)
//    //        {
//    //            if (x == 0 && y == 0 && z == 0)
//    //                continue;
                
//    //            int3 offs = int3(offset0.x * x, offset1.y * y, offset2.z * z);
//    //            iNeighbourCoords[arrIndex++] = iVoxelCoords + offs;
//    //        }
//    //    }
//    //}
    
//    int3 iNeighbourCoords[7];

//    iNeighbourCoords[0] = iVoxelCoords + offset0;
//    iNeighbourCoords[1] = iVoxelCoords + offset1;
//    iNeighbourCoords[2] = iVoxelCoords + offset2;
//    iNeighbourCoords[3] = iVoxelCoords + offset0 + offset1;
//    iNeighbourCoords[4] = iVoxelCoords + offset0 + offset2;
//    iNeighbourCoords[5] = iVoxelCoords + offset1 + offset2;
//    iNeighbourCoords[6] = iVoxelCoords + offset0 + offset1 + offset2;
    
//        uint linearIdx;
    
//    float4 arrayIrradiance[7];
//    uint neighbourVoxIdx;
    
//    uint3 neighbourCoords;
//    for (int i = 0; i < 7; i++)
//    {
//        arrayIrradiance[i] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        
//        if (any(iNeighbourCoords[i] < 0) || any(iNeighbourCoords[i] >= int3(voxelTextureDimensions)))
//            continue;
        
//        neighbourCoords = uint3(iNeighbourCoords[i]);
        
//        linearIdx = neighbourCoords.x + neighbourCoords.y * voxelTextureDimensions.x + neighbourCoords.z * voxelTextureDimensions.x * voxelTextureDimensions.y;
        
//        if (IsVoxelPresent(linearIdx))
//        {
//            uint neighbourVoxIdx = FindHashedCompactedPositionIndex(uint3(iNeighbourCoords[i])).x;
            
//            uint2 packedRadiance = gPackedRadiance[neighbourVoxIdx * 6 + index];
            
//            arrayIrradiance[i].xy = UnpackFloats16(packedRadiance.x);
//            arrayIrradiance[i].z = UnpackFloats16(packedRadiance.y).x;
            
//            arrayIrradiance[i].w = dot(arrayIrradiance[i].xyz, float3(1.0f, 1.0f, 1.0f));
//        }
//    }
    
//    uint topIndices[3] = { 0, 0, 0 };
    
//    float highestValues[3] = { -1.0f, -1.0f, -1.0f };
    
//    // Finds the combination of 3 neighbour voxels that gives the highest irradiance

//    uint IndexOfMax = -1;
//    for (i = 0; i < 7; ++i)
//    {
//        float currentIrradiance = arrayIrradiance[i].w;
        
//        uint minIndex = 0;
//        float minValue = highestValues[0];

//        for (uint j = 1; j < 3; j++)
//        {
//            if (highestValues[j] < minValue)
//            {
//                minValue = highestValues[j];
//                minIndex = j;
//            }
//        }
        
//        if (currentIrradiance > minValue)
//        {
//            highestValues[minIndex] = currentIrradiance;
//            topIndices[minIndex] = i;
//            IndexOfMax = 1;
//        }
//    }
    
//    uint temp = 0;
//    if (topIndices[0] > topIndices[1])
//    {
//        temp = topIndices[0];
//        topIndices[0] = topIndices[1];
//        topIndices[1] = temp;
//    }
//    if (topIndices[1] > topIndices[2])
//    {
//        temp = topIndices[1];
//        topIndices[1] = topIndices[2];
//        topIndices[2] = temp;
//    }
//    if (topIndices[0] > topIndices[1])
//    {
//        temp = topIndices[0];
//        topIndices[0] = topIndices[1];
//        topIndices[1] = temp;
//    }
    

//    if (IndexOfMax == -1)
//        return radiance;

//    float3 arrayIrradianceInterpolation[4];
//    arrayIrradianceInterpolation[0] = radiance;
//    arrayIrradianceInterpolation[1] = arrayIrradiance[topIndices[0]].xyz;
//    arrayIrradianceInterpolation[2] = arrayIrradiance[topIndices[1]].xyz;
//    arrayIrradianceInterpolation[3] = arrayIrradiance[topIndices[2]].xyz;

//    float3 arraySquareInterpolation[4];
//    arraySquareInterpolation[0] = voxelWorldCoords;
//    arraySquareInterpolation[1] = mul(float4(float3(iNeighbourCoords[topIndices[0]]), 1.0f), VoxelToWorld).xyz;
//    arraySquareInterpolation[2] = mul(float4(float3(iNeighbourCoords[topIndices[1]]), 1.0f), VoxelToWorld).xyz;
//    arraySquareInterpolation[3] = mul(float4(float3(iNeighbourCoords[topIndices[2]]), 1.0f), VoxelToWorld).xyz;
    
    
//    // Possible problem with those cases in which only one voxel is displaced: find out since the computed plane could be wrong

//    float3 planeNormal = normalize(cross(arraySquareInterpolation[1] - voxelWorldCoords, arraySquareInterpolation[2] - voxelWorldCoords));
//    float3 projectedFragment = pointPlaneProjection(fragmentWorldPos, planeNormal, voxelWorldCoords);

//    // Now, find out how points in arraySquareInterpolation form a square in 3D, take arraySquareInterpolation[0] as reference
//    float distanceMaximized = 0.0f;
//    float distanceSqTemp = 0.0f;
//    float3 farthest = float3(0.0f, 0.0f, 0.0f);
//    float3 neighbour0 = float3(0.0f, 0.0f, 0.0f);
//    float3 neighbour1 = float3(0.0f, 0.0f, 0.0f);

//    float3 irradianceNeighbour0 = float3(0.0f, 0.0f, 0.0f);
//    float3 irradianceNeighbour1 = float3(0.0f, 0.0f, 0.0f);
//    float3 irradianceFarthest = float3(0.0f, 0.0f, 0.0f);

//    int farthestIndex = 0;

//    for (i = 1; i < 4; ++i)
//    {
//        distanceSqTemp = distanceSq(arraySquareInterpolation[0], arraySquareInterpolation[i]);

//        if (distanceSqTemp > distanceMaximized)
//        {
//            farthest = arraySquareInterpolation[i];
//            irradianceFarthest = arrayIrradianceInterpolation[i];
//            farthestIndex = i;
//        }
//    }
    
//    switch (farthestIndex)
//    {
//        case 1:
//        {
//                neighbour0 = arraySquareInterpolation[2];
//                neighbour1 = arraySquareInterpolation[3];
//                irradianceNeighbour0 = arrayIrradianceInterpolation[2];
//                irradianceNeighbour1 = arrayIrradianceInterpolation[3];
//            }
//            break;
//        case 2:
//        {
//                neighbour0 = arraySquareInterpolation[1];
//                neighbour1 = arraySquareInterpolation[3];
//                irradianceNeighbour0 = arrayIrradianceInterpolation[1];
//                irradianceNeighbour1 = arrayIrradianceInterpolation[3];
//            }
//            break;
//        case 3:
//        {
//                neighbour0 = arraySquareInterpolation[1];
//                neighbour1 = arraySquareInterpolation[2];
//                irradianceNeighbour0 = arrayIrradianceInterpolation[1];
//                irradianceNeighbour1 = arrayIrradianceInterpolation[2];
//            }
//            break;
//    }
    
    
//    arraySquareInterpolation[0] = voxelWorldCoords;
//    arraySquareInterpolation[1] = neighbour0;
//    arraySquareInterpolation[2] = neighbour1;
//    arraySquareInterpolation[3] = farthest;

//    arrayIrradianceInterpolation[0] = radiance;
//    arrayIrradianceInterpolation[1] = irradianceNeighbour0;
//    arrayIrradianceInterpolation[2] = irradianceNeighbour1;
//    arrayIrradianceInterpolation[3] = irradianceFarthest;

//    bool arrayZeroIrradiance[4];
//    int numZeroIrradiance = 0;

//    float3 arrayNonZeroIrradiancePoint[4];
//    float3 arrayNonZeroIrradianceValue[4];

//    float3 vecOne = float3(1.0f, 1.0f, 1.0f);
//    for (i = 0; i < 4; ++i)
//    {
//        if (dot(arrayIrradianceInterpolation[i], vecOne) > 0.0f)
//        {
//            arrayNonZeroIrradiancePoint[numZeroIrradiance] = arraySquareInterpolation[i];
//            arrayNonZeroIrradianceValue[numZeroIrradiance] = arrayIrradianceInterpolation[i];
//            numZeroIrradiance++;
//        }
//    }
    
//    //if (hasBigDisplacement)
//    //{
//    //    return float3(0.0f, 0.0f, 0.0f);
//    //}
    
//    if (numZeroIrradiance == 2)
//    {
//        // In case two of the four irradiance values are zero, interpolate irradiance by projecting the fragment onto the segment
//        // between the two non-zero irradiance values
//        return approximateIrradianceTwoPoint(arrayNonZeroIrradiancePoint[0], arrayNonZeroIrradiancePoint[1],
//                                             arrayNonZeroIrradianceValue[0], arrayNonZeroIrradianceValue[1],
//                                             fragmentWorldPos);
//    }
//    else if (numZeroIrradiance == 3)
//    {
        
//        // In case one of the four points has zero irradiance value, interpolate through a triangle
//        float3 meanIrradiance = (arrayNonZeroIrradianceValue[0] + arrayNonZeroIrradianceValue[1] + arrayNonZeroIrradianceValue[2]) / 3.0f;

//        if (all(radiance == 0.0f))
//        {
//            radiance = meanIrradiance;
//        }
//        else if (all(irradianceNeighbour0 == 0.0f))
//        {
//            irradianceNeighbour0 = meanIrradiance;
//        }
//        else if (all(irradianceNeighbour1 == 0.0f))
//        {
//            irradianceNeighbour1 = meanIrradiance;
//        }
//        else if (all(irradianceFarthest == 0.0f))
//        {
//            irradianceFarthest = meanIrradiance;
//        }
//    }
    
//    float3 projectionNeighbour0 = pointSegmentProjection(voxelWorldCoords, neighbour0, projectedFragment);
//    float3 projectionNeighbour1 = pointSegmentProjection(voxelWorldCoords, neighbour1, projectedFragment);
//    float normalizedCoordinatesX = distance(voxelWorldCoords, projectionNeighbour0) / distance(voxelWorldCoords, neighbour0);
//    float normalizedCoordinatesY = distance(voxelWorldCoords, projectionNeighbour1) / distance(voxelWorldCoords, neighbour1);

    
    
//    float3 lerpX1 = lerp(radiance, irradianceNeighbour0, normalizedCoordinatesX);
//    float3 lerpX2 = lerp(irradianceNeighbour1, irradianceFarthest, normalizedCoordinatesX);
//    float3 result = lerp(lerpX1, lerpX2, normalizedCoordinatesY);
    
//    return result;
//}


float3 InterpolateIrradiance(uint3 voxelCoords, float3 fragmentWorldPos, uint index, bool isDisplaced)
{
    int3 arrayCombinationIndex[35] =
    {
        int3(0, 1, 2),
    int3(0, 1, 3),
    int3(0, 1, 4),
    int3(0, 1, 5),
    int3(0, 1, 6),
    int3(0, 2, 3),
    int3(0, 2, 4),
    int3(0, 2, 5),
    int3(0, 2, 6),
    int3(0, 3, 4),
    int3(0, 3, 5),
    int3(0, 3, 6),
    int3(0, 4, 5),
    int3(0, 4, 6),
    int3(0, 5, 6),
    int3(1, 2, 3),
    int3(1, 2, 4),
    int3(1, 2, 5),
    int3(1, 2, 6),
    int3(1, 3, 4),
    int3(1, 3, 5),
    int3(1, 3, 6),
    int3(1, 4, 5),
    int3(1, 4, 6),
    int3(1, 5, 6),
    int3(2, 3, 4),
    int3(2, 3, 5),
    int3(2, 3, 6),
    int3(2, 4, 5),
    int3(2, 4, 6),
    int3(2, 5, 6),
    int3(3, 4, 5),
    int3(3, 4, 6),
    int3(3, 5, 6),
    int3(4, 5, 6)
    };

    
    float3 voxelWorldCoords = mul(float4(float3(voxelCoords), 1.0f), VoxelToWorld).xyz;
    float3 voxelCenterToFragment = normalize(fragmentWorldPos - voxelWorldCoords);

    int3 iVoxelCoords = int3(voxelCoords);
    
    bool hasBigDisplacement = isDisplaced && (distance(fragmentWorldPos, voxelWorldCoords) > voxelCellSize.x / 2.0f);

    
    int3 offset0 = int3(sign(voxelCenterToFragment.x), 0, 0);
    int3 offset1 = int3(0, sign(voxelCenterToFragment.y), 0);
    int3 offset2 = int3(0, 0, sign(voxelCenterToFragment.z));
    
    bool isNotAligned = false;
    
    uint voxelIdx = FindHashedCompactedPositionIndex(voxelCoords).x;

    uint faceIdx = voxelIdx * 6 + index;
    
    
    uint2 packedRadiance = gPackedRadiance[faceIdx];
    
    float3 radiance = float3(0.0f, 0.0f, 0.0f);
    
    radiance.xy = UnpackFloats16(packedRadiance.x);
    radiance.z = UnpackFloats16(packedRadiance.y).x;

    int3 iNeighbourCoords[7];

    iNeighbourCoords[0] = iVoxelCoords + offset0;
    iNeighbourCoords[1] = iVoxelCoords + offset1;
    iNeighbourCoords[2] = iVoxelCoords + offset2;
    iNeighbourCoords[3] = iVoxelCoords + offset0 + offset1;
    iNeighbourCoords[4] = iVoxelCoords + offset0 + offset2;
    iNeighbourCoords[5] = iVoxelCoords + offset1 + offset2;
    iNeighbourCoords[6] = iVoxelCoords + offset0 + offset1 + offset2;
    
    uint linearIdx;
    
    float3 arrayIrradiance[7];
    uint neighbourVoxIdx;
    
    uint3 neighbourCoords;
    for (int i = 0; i < 7; i++)
    {
        arrayIrradiance[i] = float3(0.0f, 0.0f, 0.0f);
        
        if (any(iNeighbourCoords[i] < 0) || any(iNeighbourCoords[i] >= int3(voxelTextureDimensions)))
            continue;
        
        neighbourCoords = uint3(iNeighbourCoords[i]);
        
        linearIdx = neighbourCoords.x + neighbourCoords.y * voxelTextureDimensions.x + neighbourCoords.z * voxelTextureDimensions.x * voxelTextureDimensions.y;
        
        if (IsVoxelPresent(linearIdx))
        {
            uint neighbourVoxIdx = FindHashedCompactedPositionIndex(uint3(iNeighbourCoords[i])).x;
            
            uint2 packedRadiance = gPackedRadiance[neighbourVoxIdx * 6 + index];
            
            arrayIrradiance[i].xy = UnpackFloats16(packedRadiance.x);
            arrayIrradiance[i].z = UnpackFloats16(packedRadiance.y).x;
        }
    }
    
    // Finds the combination of 3 neighbour voxels that gives the highest irradiance
    float currentIrradiance = 0.0f;
    float maxIrradiance = 0.0f;
    uint IndexOfMax = -1;
    float3 vecOne = float3(1.0f, 1.0f, 1.0f);
    for (i = 0; i < 35; ++i)
    {
        currentIrradiance = dot(arrayIrradiance[arrayCombinationIndex[i].x], vecOne) + dot(arrayIrradiance[arrayCombinationIndex[i].y], vecOne) + dot(arrayIrradiance[arrayCombinationIndex[i].z], vecOne);

        if (currentIrradiance > maxIrradiance)
        {
            maxIrradiance = currentIrradiance;
            IndexOfMax = i;
        }
    }
    
    float3 result = float3(0.0f, 0.0f, 0.0f);
    
    if (IndexOfMax == -1)
    {
    
        result = radiance;
    }
    else
    {
    
        uint index0 = arrayCombinationIndex[IndexOfMax].x;
        uint index1 = arrayCombinationIndex[IndexOfMax].y;
        uint index2 = arrayCombinationIndex[IndexOfMax].z;

        float3 arrayIrradianceInterpolation[4];
        arrayIrradianceInterpolation[0] = radiance;
        arrayIrradianceInterpolation[1] = arrayIrradiance[index0];
        arrayIrradianceInterpolation[2] = arrayIrradiance[index1];
        arrayIrradianceInterpolation[3] = arrayIrradiance[index2];

        float3 arraySquareInterpolation[4];
        arraySquareInterpolation[0] = voxelWorldCoords;
        arraySquareInterpolation[1] = mul(float4(float3(iNeighbourCoords[index0]), 1.0f), VoxelToWorld).xyz;
        arraySquareInterpolation[2] = mul(float4(float3(iNeighbourCoords[index1]), 1.0f), VoxelToWorld).xyz;
        arraySquareInterpolation[3] = mul(float4(float3(iNeighbourCoords[index2]), 1.0f), VoxelToWorld).xyz;
    
    
    // Possible problem with those cases in which only one voxel is displaced: find out since the computed plane could be wrong

        float3 planeNormal = normalize(cross(arraySquareInterpolation[1] - voxelWorldCoords, arraySquareInterpolation[2] - voxelWorldCoords));
        float3 projectedFragment = pointPlaneProjection(fragmentWorldPos, planeNormal, voxelWorldCoords);

    // Now, find out how points in arraySquareInterpolation form a square in 3D, take arraySquareInterpolation[0] as reference
        float distanceMaximized = 0.0f;
        float distanceSqTemp = 0.0f;
        float3 farthest = float3(0.0f, 0.0f, 0.0f);
        float3 neighbour0 = float3(0.0f, 0.0f, 0.0f);
        float3 neighbour1 = float3(0.0f, 0.0f, 0.0f);

        float3 irradianceNeighbour0 = float3(0.0f, 0.0f, 0.0f);
        float3 irradianceNeighbour1 = float3(0.0f, 0.0f, 0.0f);
        float3 irradianceFarthest = float3(0.0f, 0.0f, 0.0f);

        int farthestIndex = 0;

        for (i = 1; i < 4; ++i)
        {
            distanceSqTemp = distanceSq(arraySquareInterpolation[0], arraySquareInterpolation[i]);

            if (distanceSqTemp > distanceMaximized)
            {
                farthest = arraySquareInterpolation[i];
                irradianceFarthest = arrayIrradianceInterpolation[i];
                farthestIndex = i;
            }
        }
    
        switch (farthestIndex)
        {
            case 1:
        {
                    neighbour0 = arraySquareInterpolation[2];
                    neighbour1 = arraySquareInterpolation[3];
                    irradianceNeighbour0 = arrayIrradianceInterpolation[2];
                    irradianceNeighbour1 = arrayIrradianceInterpolation[3];
                }
                break;
            case 2:
        {
                    neighbour0 = arraySquareInterpolation[1];
                    neighbour1 = arraySquareInterpolation[3];
                    irradianceNeighbour0 = arrayIrradianceInterpolation[1];
                    irradianceNeighbour1 = arrayIrradianceInterpolation[3];
                }
                break;
            case 3:
        {
                    neighbour0 = arraySquareInterpolation[1];
                    neighbour1 = arraySquareInterpolation[2];
                    irradianceNeighbour0 = arrayIrradianceInterpolation[1];
                    irradianceNeighbour1 = arrayIrradianceInterpolation[2];
                }
                break;
        }
    
    
        arraySquareInterpolation[0] = voxelWorldCoords;
        arraySquareInterpolation[1] = neighbour0;
        arraySquareInterpolation[2] = neighbour1;
        arraySquareInterpolation[3] = farthest;

        arrayIrradianceInterpolation[0] = radiance;
        arrayIrradianceInterpolation[1] = irradianceNeighbour0;
        arrayIrradianceInterpolation[2] = irradianceNeighbour1;
        arrayIrradianceInterpolation[3] = irradianceFarthest;

        bool arrayZeroIrradiance[4];
        int numZeroIrradiance = 0;

        float3 arrayNonZeroIrradiancePoint[4];
        float3 arrayNonZeroIrradianceValue[4];

    
        for (i = 0; i < 4; ++i)
        {
            if (dot(arrayIrradianceInterpolation[i], vecOne) > 0.0f)
            {
                arrayNonZeroIrradiancePoint[numZeroIrradiance] = arraySquareInterpolation[i];
                arrayNonZeroIrradianceValue[numZeroIrradiance] = arrayIrradianceInterpolation[i];
                numZeroIrradiance++;
            }
        }
    
    //if (hasBigDisplacement)
    //{
    //    return float3(0.0f, 0.0f, 0.0f);
    //}
    
        if (numZeroIrradiance == 2)
        {
        // In case two of the four irradiance values are zero, interpolate irradiance by projecting the fragment onto the segment
        // between the two non-zero irradiance values
            result = approximateIrradianceTwoPoint(arrayNonZeroIrradiancePoint[0], arrayNonZeroIrradiancePoint[1],
                                             arrayNonZeroIrradianceValue[0], arrayNonZeroIrradianceValue[1],
                                             fragmentWorldPos);
        }
        else
        {
            if (numZeroIrradiance == 3)
            {
            
        // In case one of the four points has zero irradiance value, interpolate through a triangle
                float3 meanIrradiance = (arrayNonZeroIrradianceValue[0] + arrayNonZeroIrradianceValue[1] + arrayNonZeroIrradianceValue[2]) / 3.0f;

                if (all(radiance == 0.0f))
                {
                    radiance = meanIrradiance;
                }
                else if (all(irradianceNeighbour0 == 0.0f))
                {
                    irradianceNeighbour0 = meanIrradiance;
                }
                else if (all(irradianceNeighbour1 == 0.0f))
                {
                    irradianceNeighbour1 = meanIrradiance;
                }
                else if (all(irradianceFarthest == 0.0f))
                {
                    irradianceFarthest = meanIrradiance;
                }
            }
    
            float3 projectionNeighbour0 = pointSegmentProjection(voxelWorldCoords, neighbour0, projectedFragment);
            float3 projectionNeighbour1 = pointSegmentProjection(voxelWorldCoords, neighbour1, projectedFragment);
            float normalizedCoordinatesX = distance(voxelWorldCoords, projectionNeighbour0) / distance(voxelWorldCoords, neighbour0);
            float normalizedCoordinatesY = distance(voxelWorldCoords, projectionNeighbour1) / distance(voxelWorldCoords, neighbour1);

    
    
            float3 lerpX1 = lerp(radiance, irradianceNeighbour0, normalizedCoordinatesX);
            float3 lerpX2 = lerp(irradianceNeighbour1, irradianceFarthest, normalizedCoordinatesX);
            float3 lerpyY = lerp(lerpX1, lerpX2, normalizedCoordinatesY);
    
            result = lerpyY;
        }
    }
    
    return result;
}

PSOut PS(VertexOutPosTex pIn)
{
    
    float4 diffuse = gBufferDiffuse.Sample(gSampler, pIn.Tex);
    float4 worldPos = gBufferaWorld.Sample(gSampler, pIn.Tex);
    
    float2 normalPacked = gBufferNormal.Sample(gSampler, pIn.Tex).rg;
    float3 normal = UnpackNormal(normalPacked);
    float4 RMA = gBufferMetallicRoughnessAO.Sample(gSampler, pIn.Tex);
        
    // w coordinate is used to keep track of geometry. If there is no geometry, the w value is 1.0f so we can discard the pixel
    if (RMA.w >= 1.0f)
        discard;
     
    
    PBRMaterial material = GetPBRMaterial(gMaterials[worldPos.w]);
    
    float3 V = normalize(camera.EyePos - worldPos.xyz);
    

    diffuse = diffuse * material.baseColor;
    
    // glTF stores roguhness in the G channel, metallic in the B channel and AO in the R channel
    float roughness = material.roughness * RMA.g;
    float metallic = material.metallic * RMA.b;
    float occlusion = RMA.r;

    
    SurfaceData surfData;
    surfData.N = normal;
    surfData.V = V;
    surfData.NdotV = saturate(dot(surfData.N, surfData.V));
    surfData.c_diff = lerp(diffuse.rgb, float3(0, 0, 0), metallic) * occlusion;
    surfData.c_spec = lerp(kDielectricSpecular, diffuse.rgb, metallic) * occlusion;
    
    float3 lRes = float3(0.0f, 0.0f, 0.0f);

   
    for (int i = 0; i < commons.NumLights; i++)
    {

        Light light = gLights[i];
        
        float shadowFactor = 1.0f;
        
        if (light.castShadows)
        {
            float4 shadowPosH = mul(float4(worldPos.xyz, 1.0f), light.shadowMatrix);
            shadowFactor = CalcShadowFactor(gShadowMap, shadowPosH);
        }
        
        light.color *= shadowFactor;

        lRes += PBRDirectionalLight(light, surfData, roughness);
    }
    
    PSOut psOut;
    psOut.DeferredTexture = float4(lRes, diffuse.a);
    
    if (commons.UseRTGI == 0)
    {
        psOut.RadianceTexture = float4(surfData.c_diff * 0.13f, 1.0f);
    }
    else
    {
        float3 worldCoord = worldPos.xyz;
        uint3 voxelCoord = uint3(mul(float4(worldCoord, 1.0f), WorldToVoxel).xyz);
        uint linearCoord = voxelCoord.x + voxelCoord.y * voxelTextureDimensions.x + voxelCoord.z * voxelTextureDimensions.x * voxelTextureDimensions.y;
    
        float4 radiance = float4(0.0f, 0.0f, 0.0f, 0.0f);
        bool found = IsVoxelPresent(linearCoord);
        bool displaced = false;
        uint3 secondClosest = uint3(UINT_MAX, UINT_MAX, UINT_MAX);
        if (!found)
        {
            uint3 newCoords;
            found = FindClosestOccupiedVoxel(worldCoord, voxelCoord, newCoords);
            voxelCoord = newCoords;
            displaced = true;
        }
        if (found)
        {
            uint index = FindMostAlignedDirection(normal);
            
            float3 voxelWorldCoords = mul(float4(float3(voxelCoord), 1.0f), VoxelToWorld).xyz;
            bool hasBigDisplacement = displaced && (distance(worldCoord, voxelWorldCoords) > voxelCellSize.x);

            radiance.xyz = InterpolateIrradiance(voxelCoord, worldCoord, index, displaced);
            radiance.w = hasBigDisplacement ? 1.0f : 0.0f;
        }
        
        psOut.RadianceTexture = radiance;
    }
   
    
    return psOut;
}