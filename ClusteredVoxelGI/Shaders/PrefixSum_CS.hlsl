#include "VoxelUtils.hlsli"

cbuffer cbPrefixSum : register(b0)
{
    uint gCurrentPhase;
    uint gCurrentStep;
    uint gPrefixSumBufferSize;
    uint gElementsPerThread;
    
    uint gNumStepsSweepDown;
    uint gNumElementsBase;
    uint gNumElementsLevel0;
    uint gNumElementsLevel1;
    
    uint gNumElementsLevel2;
    uint gNumElementsLevel3;
    float pad0;
    float pad1;
    
    uint3 gVoxelGridDimension;
    float pad2;
}

RWStructuredBuffer<FragmentData> gFragmentDataBuffer : register(u0, space0);
RWStructuredBuffer<uint> gNextIndexBuffer : register(u1, space0);

RWStructuredBuffer<uint> gVoxelIndicesBuffer : register(u2, space0);



RWStructuredBuffer<uint> gIndirectionRankBuffer : register(u0, space1);
RWStructuredBuffer<uint> gIndirectionIndexBuffer : register(u1, space1);

RWStructuredBuffer<uint> gVoxelIndicesCompacted : register(u2, space1);
RWStructuredBuffer<uint> gVoxelHashesCompacted : register(u3, space1);

RWStructuredBuffer<uint> gPrefixSumBuffer : register(u4, space1);

[numthreads(128, 1, 1)]
void CS(uint3 GroupID : SV_GroupID, uint GroupThreadIndex : SV_GroupIndex)
{
    
    uint index = GroupID.x * 128 + GroupThreadIndex;

    if (gCurrentPhase == 0)
    {
    
        uint arrayMaxIndex[5];
        arrayMaxIndex[0] = gNumElementsBase;
        arrayMaxIndex[1] = gNumElementsLevel0;
        arrayMaxIndex[2] = gNumElementsLevel1;
        arrayMaxIndex[3] = gNumElementsLevel2;
        arrayMaxIndex[4] = gNumElementsLevel3;

        uint maxIndex = arrayMaxIndex[gCurrentStep];

        if (index >= maxIndex)
        {
            return;
        }

        uint arrayInitialIndex[5];
        arrayInitialIndex[0] = gElementsPerThread * index ;
        arrayInitialIndex[1] = gElementsPerThread * index ;
        arrayInitialIndex[2] = gElementsPerThread * index + gNumElementsBase;
        arrayInitialIndex[3] = gElementsPerThread * index + gNumElementsBase + gNumElementsLevel0;
        arrayInitialIndex[4] = gElementsPerThread * index + gNumElementsBase + gNumElementsLevel0 + gNumElementsLevel1;

        const uint initialIndex = arrayInitialIndex[gCurrentStep];

        uint arrayFinalIndex[5];
        arrayFinalIndex[0] = initialIndex + gElementsPerThread;
        arrayFinalIndex[1] = initialIndex + min(gElementsPerThread, gNumElementsBase);
        arrayFinalIndex[2] = initialIndex + min(gElementsPerThread, gNumElementsLevel0);
        arrayFinalIndex[3] = initialIndex + min(gElementsPerThread, gNumElementsLevel1);
        arrayFinalIndex[4] = initialIndex + min(gElementsPerThread, gNumElementsLevel2);
        
        const uint finalIndex = arrayFinalIndex[gCurrentStep];

        uint arrayThreadWriteIndex[5];
        arrayThreadWriteIndex[0] = index;
        arrayThreadWriteIndex[1] = index + gNumElementsBase;
        arrayThreadWriteIndex[2] = index + gNumElementsBase + gNumElementsLevel0;
        arrayThreadWriteIndex[3] = index + gNumElementsBase + gNumElementsLevel0 + gNumElementsLevel1;
        arrayThreadWriteIndex[4] = index + gNumElementsBase + gNumElementsLevel0 + gNumElementsLevel1 + gNumElementsLevel2;

        const uint threadWriteIndex = arrayThreadWriteIndex[gCurrentStep];
        uint nonNullCounter = 0;
    
        if (gCurrentStep == 0)
        {
            for (uint i = initialIndex; i < finalIndex; i++)
            {
                if (gVoxelIndicesBuffer[i] != UINT_MAX)
                {
                    nonNullCounter += 1;
                }
            }
        }
        else
        {
            for (uint i = initialIndex; i < finalIndex; i++)
            {
                nonNullCounter += gPrefixSumBuffer[i];
            }
        }

        gPrefixSumBuffer[threadWriteIndex] = nonNullCounter;
    }
    else if (gCurrentPhase == 1)
    {
        uint arrayMaxIndex[5];
        arrayMaxIndex[0] = uint(ceil(float(gNumElementsBase) / float(gElementsPerThread)));
        arrayMaxIndex[1] = uint(ceil(float(gNumElementsLevel0) / float(gElementsPerThread)));
        arrayMaxIndex[2] = uint(ceil(float(gNumElementsLevel1) / float(gElementsPerThread)));
        arrayMaxIndex[3] = uint(ceil(float(gNumElementsLevel2) / float(gElementsPerThread)));
        arrayMaxIndex[4] = uint(ceil(float(gNumElementsLevel3) / float(gElementsPerThread)));

        uint maxIndex = arrayMaxIndex[gCurrentStep];

        if (index >= maxIndex)
        {
            return;
        }

        
        uint arrayInitialIndex[5];
        arrayInitialIndex[0] = gElementsPerThread * index;
        arrayInitialIndex[1] = gElementsPerThread * index + gNumElementsBase;
        arrayInitialIndex[2] = gElementsPerThread * index + gNumElementsBase + gNumElementsLevel0;
        arrayInitialIndex[3] = gElementsPerThread * index + gNumElementsBase + gNumElementsLevel0 + gNumElementsLevel1;
        arrayInitialIndex[4] = gElementsPerThread * index + gNumElementsBase + gNumElementsLevel0 + gNumElementsLevel1 + gNumElementsLevel2;

        uint initialIndex = arrayInitialIndex[gCurrentStep];

        uint arrayFinalIndex[5];
        arrayFinalIndex[0] = initialIndex + min(gElementsPerThread, gNumElementsBase);
        arrayFinalIndex[1] = initialIndex + min(gElementsPerThread, gNumElementsLevel0);
        arrayFinalIndex[2] = initialIndex + min(gElementsPerThread, gNumElementsLevel1);
        arrayFinalIndex[3] = initialIndex + min(gElementsPerThread, gNumElementsLevel2);
        arrayFinalIndex[4] = initialIndex + min(gElementsPerThread, gNumElementsLevel3);


        uint arrayMaxBaseIndex[5];
        arrayMaxBaseIndex[0] = gNumElementsBase;
        arrayMaxBaseIndex[1] = gNumElementsBase + gNumElementsLevel0;
        arrayMaxBaseIndex[2] = gNumElementsBase + gNumElementsLevel0 + gNumElementsLevel1;
        arrayMaxBaseIndex[3] = gNumElementsBase + gNumElementsLevel0 + gNumElementsLevel1 + gNumElementsLevel2;
        arrayMaxBaseIndex[4] = gNumElementsBase + gNumElementsLevel0 + gNumElementsLevel1 + gNumElementsLevel2 + gNumElementsLevel3;
        
        const uint maxBaseIndex = arrayMaxBaseIndex[gCurrentStep];
        
        const uint finalIndex = min(arrayFinalIndex[gCurrentStep], maxBaseIndex);

        uint previousValue;
        
		// Don't add the offset from the previous step
        if ((gCurrentStep + 1) == gNumStepsSweepDown)
        {
            uint accumulated = gPrefixSumBuffer[initialIndex];
            gPrefixSumBuffer[initialIndex] = 0;

            for (uint i = initialIndex + 1; i < finalIndex; ++i)
            {
                previousValue = gPrefixSumBuffer[i];
                gPrefixSumBuffer[i] = accumulated;
                accumulated += previousValue;
            }
        }
        else
        {
			// Not the first iteration of the algorithm, and not the last, when the results in the first myMaterialData.numElementBase
			// elements of prefixSumPlanarBuffer are used to build the final compacted buffer from the initial voxelFirstIndexBuffer
            uint arrayOffsetIndex[5];
            arrayOffsetIndex[0] = index + gNumElementsBase;
            arrayOffsetIndex[1] = index + gNumElementsBase + gNumElementsLevel0;
            arrayOffsetIndex[2] = index + gNumElementsBase + gNumElementsLevel0 + gNumElementsLevel1;
            arrayOffsetIndex[3] = index + gNumElementsBase + gNumElementsLevel0 + gNumElementsLevel1 + gNumElementsLevel2;
            arrayOffsetIndex[4] = index + gNumElementsBase + gNumElementsLevel0 + gNumElementsLevel1 + gNumElementsLevel2 + gNumElementsLevel3;

            uint previousStepOffsetIndex = arrayOffsetIndex[gCurrentStep];
            uint offset = gPrefixSumBuffer[previousStepOffsetIndex];
            uint accumulated = 0;

            for (uint i = initialIndex; i < finalIndex; ++i)
            {
                previousValue = gPrefixSumBuffer[i];
                gPrefixSumBuffer[i] = offset + accumulated;
                accumulated += previousValue;
            }
        }

    }
    else if (gCurrentPhase == 2)
    {
		// Last step of the algorithm: loop through the original buffer voxelFirstIndexCompacted and put any element different from
		// the default alue (given by maxValue) at the proper index taking the swept down index from voxelFirstIndex
        uint maxIndex = gNumElementsBase;

        if (index >= maxIndex)
        {
            return;
        }

        const uint initialIndex = index * gElementsPerThread;
        const uint finalIndex = initialIndex + gElementsPerThread;
        const uint offset = gPrefixSumBuffer[index];

        uint counter = 0;
        for (uint i = initialIndex; i < finalIndex; ++i)
        {
            if (gVoxelIndicesBuffer[i] != UINT_MAX)
            {
                gVoxelIndicesCompacted[offset + counter] = gVoxelIndicesBuffer[i];
                gVoxelHashesCompacted[offset + counter] = i;
                uint3 result = GetVoxelPosition(i, gVoxelGridDimension);

                uint voxelizationSize = gVoxelGridDimension.z;
                uint indirectionIndex = voxelizationSize * result.z + result.y;

                InterlockedAdd(gIndirectionRankBuffer[indirectionIndex], 1);
                InterlockedMin(gIndirectionIndexBuffer[indirectionIndex], offset + counter);

                counter++;
            }
        }
    }
}