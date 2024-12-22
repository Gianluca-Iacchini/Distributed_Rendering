#include "PrefixSumVoxels.h"
#include "DX12Lib/pch.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"
#include "VoxelizeScene.h"
#include "../Data/Shaders/Include/PrefixSum_CS.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;
using namespace VOX;

CVGI::PrefixSumVoxels::PrefixSumVoxels(std::shared_ptr<VOX::TechniqueData> data)
{
	m_bufferManager = std::make_shared<BufferManager>();
	data->SetBufferManager(Name, m_bufferManager);
	m_data = data;
}

void CVGI::PrefixSumVoxels::InitializeBuffers(DX12Lib::ComputeContext& context)
{

	DirectX::XMUINT3 voxelGridSize = m_data->GetVoxelGridSize();

	// Indirection Rank Buffer (u0)
	m_bufferManager->AddStructuredBuffer(voxelGridSize.y * voxelGridSize.z, sizeof(UINT32));

	// Indirection Index Buffer (u1)
	UINT indIndBuffIdx = m_bufferManager->AddStructuredBuffer(voxelGridSize.y * voxelGridSize.z, sizeof(UINT32));

	UploadBuffer uploadBuffer;
	uploadBuffer.Create(voxelGridSize.y * voxelGridSize.z * sizeof(UINT32));

	void* mappedData = uploadBuffer.Map();

	for (UINT32 i = 0; i < voxelGridSize.y * voxelGridSize.z; i++)
	{
		((UINT32*)mappedData)[i] = UINT_MAX;
	}


	DX12Lib::GPUBuffer& indIndBuff = m_bufferManager->GetBuffer(indIndBuffIdx);
	// Not using CommandContext.CopyBuffer because upload buffer should not be transitioned from the GENERIC_READ state
	context.TransitionResource(indIndBuff, D3D12_RESOURCE_STATE_COPY_DEST, true);
	context.m_commandList->Get()->CopyResource(indIndBuff.Get(), uploadBuffer.Get());


	ComputePrefixSumVariables();

	// Voxel Index Compacted Buffer (u2)
	// Will be resized later
	m_bufferManager->AddStructuredBuffer(1, sizeof(UINT32));

	// Compact hash buffer (u3)
	// Will be resized later
	m_bufferManager->AddStructuredBuffer(1, sizeof(UINT32));

	// Prefix Sum Buffer (u4)
	m_bufferManager->AddStructuredBuffer(m_prefixBufferSize, sizeof(UINT32));

	m_bufferManager->AllocateBuffers();

	DXLIB_INFO("Number of steps reduce: {0}", m_reduceStepCount);
	DXLIB_INFO("Planar buffer size: {0}", m_prefixBufferSize);



	context.Flush(true);

	uploadBuffer.Unmap();
}



void CVGI::PrefixSumVoxels::ComputePrefixSumVariables()
{
	v_prefixBufferSizeForStep.resize(5); // Up to four levels needed to the prefix parallel sum
	memset(v_prefixBufferSizeForStep.data(), 0, v_prefixBufferSizeForStep.size() * size_t(sizeof(UINT32)));

	DirectX::XMUINT3 voxelGridSize = m_data->GetVoxelGridSize();
	UINT voxelLinearSize = voxelGridSize.x * voxelGridSize.y * voxelGridSize.z;

	v_prefixBufferSizeForStep[0] = (voxelLinearSize) / ELEMENTS_PER_THREAD;

	m_prefixBufferSize = 0;
	// The maximum number of elements in bufferHistogramBuffer is 536862720,
	// given that each thread processes 128 elements of bufferHistogramBuffer in the initial pass, up to
	// four levels are needed to complete the algorithm.
	m_prefixBufferSize += UINT32(v_prefixBufferSizeForStep[0]);
	float prefixSumNumElemenCurrentStep = float(v_prefixBufferSizeForStep[0]);
	bool stop = ((prefixSumNumElemenCurrentStep / (float)ELEMENTS_PER_THREAD) <= 1.0f);

	m_reduceStepCount = 1;



	while (!stop)
	{
		prefixSumNumElemenCurrentStep = ceilf(prefixSumNumElemenCurrentStep / ELEMENTS_PER_THREAD);
		v_prefixBufferSizeForStep[m_reduceStepCount] = UINT32(prefixSumNumElemenCurrentStep);
		m_prefixBufferSize += UINT32(prefixSumNumElemenCurrentStep);

		stop = (prefixSumNumElemenCurrentStep <= 1.0f);
		m_reduceStepCount++;
	}

	m_downSweepStepCount = m_reduceStepCount;

	UINT32 numElement = UINT32(v_prefixBufferSizeForStep.size());

	m_firstSetIsSingleElement = false;

	for (UINT32 i = 0; i < numElement; i++)
	{
		if (v_prefixBufferSizeForStep[numElement - 1 - i] > 1)
		{
			if ((i > 0) && (v_prefixBufferSizeForStep[numElement - i] == 1))
			{
				m_firstSetIsSingleElement = true;
				m_downSweepStepCount -= 1;
			}
			break;
		}
	}


	m_cbCompactBuffer.VoxelGridSize = m_data->GetVoxelGridSize();
	m_cbCompactBuffer.NumElementsBase = v_prefixBufferSizeForStep[0];
	m_cbCompactBuffer.NumElementsLevel0 = v_prefixBufferSizeForStep[1];
	m_cbCompactBuffer.NumElementsLevel1 = v_prefixBufferSizeForStep[2];
	m_cbCompactBuffer.NumElementsLevel2 = v_prefixBufferSizeForStep[3];
	m_cbCompactBuffer.NumElementsLevel3 = v_prefixBufferSizeForStep[4];

	m_cbCompactBuffer.CompactBufferSize = m_prefixBufferSize;
	m_cbCompactBuffer.CurrentStep = 0;
	m_cbCompactBuffer.CurrentPhase = 0;
	m_cbCompactBuffer.NumElementsSweepDown = m_downSweepStepCount;
}

void CVGI::PrefixSumVoxels::PerformTechnique(DX12Lib::ComputeContext& context)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 0), Name.c_str());


	m_cbCompactBuffer.ElementsPerThread = ELEMENTS_PER_THREAD;

	DirectX::XMUINT3 groupSize = DirectX::XMUINT3(1, 1, 1);

	// Reduce
	for (m_currentStep = 0; m_currentStep < m_reduceStepCount; m_currentStep++)
	{
		groupSize.x = v_prefixBufferSizeForStep[m_currentStep];
		TechniquePass(context, groupSize);
		context.Flush();
	}

	UINT32 prefixSumValue = 0;
	{
		UINT32* prefixBufferData = m_bufferManager->ReadFromBuffer<UINT32*>(context, (UINT)PrefixSumBufferType::PrefixSum);
		prefixSumValue = prefixBufferData[m_prefixBufferSize - 1];
	}
	DXLIB_CORE_INFO("Prefix sum value after reduce: {0}", prefixSumValue);

	m_bufferManager->ResizeBuffer((UINT)PrefixSumBufferType::CompactedVoxelIndex, prefixSumValue);
	m_bufferManager->ResizeBuffer((UINT)PrefixSumBufferType::CompactedHashedBuffer, prefixSumValue);

	m_cmpIdxBuffer.resize(prefixSumValue);
	m_cmpHshBuffer.resize(prefixSumValue);

	// Subtracting one to m_currentStep due to the increment in the last iteration of the for loop
	m_currentStep--;




	PIXSetMarker(context.m_commandList->Get(), PIX_COLOR(0, 128, 0), L"DownSweepPass");
	m_currentPhase = 1;


	// If the first set is a single element, we start the sweep down from the second level.
	if (m_firstSetIsSingleElement)
	{
		m_currentStep--;
	}



	// Sweep down; we start from the second-last level and go down to the first level
	while ((m_currentStep + 1) > 0)
	{
		groupSize.x = UINT(ceilf(float(v_prefixBufferSizeForStep[m_currentStep]) / float(ELEMENTS_PER_THREAD)));
		TechniquePass(context, groupSize);
		m_currentStep--;
		context.Flush();
	}

	PIXSetMarker(context.m_commandList->Get(), PIX_COLOR(0, 0, 128), L"Copy Pass");
	m_currentPhase = 2;

	{
		auto val = m_bufferManager->ReadFromBuffer<UINT32*>(context, (UINT)PrefixSumBufferType::PrefixSum);
		prefixSumValue = val[m_prefixBufferSize - 1];
	}

	DXLIB_CORE_INFO("Prefix sum value after sweep: {0}", prefixSumValue);

	groupSize.x = v_prefixBufferSizeForStep[0];
	TechniquePass(context, groupSize);

	{
		auto val = m_bufferManager->ReadFromBuffer<UINT32*>(context, (UINT)PrefixSumBufferType::PrefixSum);
		prefixSumValue = val[m_prefixBufferSize - 1];
	}
	DXLIB_CORE_INFO("Prefix sum value after copy: {0}", prefixSumValue);

	DirectX::XMUINT3 voxelGridSize = m_data->GetVoxelGridSize();

	UINT32 indRankSize = voxelGridSize.y * voxelGridSize.z;
	UINT32 indIndexSize = voxelGridSize.x * voxelGridSize.y;


	m_indIdxBuffer.resize(indIndexSize);
	m_indRnkBuffer.resize(indRankSize);
	
	
	UINT32* val = m_bufferManager->ReadFromBuffer<UINT32*>(context, (UINT)PrefixSumBufferType::IndirectionRankBuffer);
	memcpy(m_indRnkBuffer.data(), val, indRankSize * sizeof(UINT32));
	
	val = m_bufferManager->ReadFromBuffer<UINT32*>(context, (UINT)PrefixSumBufferType::IndirectionIndexBuffer);
	memcpy(m_indIdxBuffer.data(), val, indIndexSize * sizeof(UINT32));

	val = m_bufferManager->ReadFromBuffer<UINT32*>(context, (UINT)PrefixSumBufferType::CompactedVoxelIndex);
	memcpy(m_cmpIdxBuffer.data(), val, prefixSumValue * sizeof(UINT32));


	val = m_bufferManager->ReadFromBuffer<UINT32*>(context, (UINT)PrefixSumBufferType::CompactedHashedBuffer);
	memcpy(m_cmpHshBuffer.data(), val, prefixSumValue * sizeof(UINT32));
	

	PIXEndEvent(context.m_commandList->Get());

	context.Flush();
}

void CVGI::PrefixSumVoxels::TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize)
{
	m_cbCompactBuffer.CurrentStep = m_currentStep;
	m_cbCompactBuffer.CurrentPhase = m_currentPhase;

	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(m_techniquePSO.get());

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)CompactBufferRootSignature::PrefixSumCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbCompactBuffer).GpuAddress());

	auto& voxelBufferManager = m_data->GetBufferManager(VoxelizeScene::Name);

	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	voxelBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)CompactBufferRootSignature::VoxelizeUAVTable, voxelBufferManager.GetUAVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)CompactBufferRootSignature::StreamCompactionUAVTable, m_bufferManager->GetUAVHandle());

	// Align to 127 because we have 127 threads per group
	UINT elementSize = (groupSize.x + 127) / 128;
	context.Dispatch(elementSize, 1, 1);
}

void CVGI::PrefixSumVoxels::DeleteTemporaryBuffers()
{
	m_bufferManager->RemoveBuffer((UINT)PrefixSumBufferType::PrefixSum);
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::PrefixSumVoxels::BuildRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> voxelComputeRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)CompactBufferRootSignature::Count, 1);
	voxelComputeRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*voxelComputeRootSignature)[(UINT)CompactBufferRootSignature::PrefixSumCBV].InitAsConstantBuffer(0);
	(*voxelComputeRootSignature)[(UINT)CompactBufferRootSignature::VoxelizeUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 4);
	(*voxelComputeRootSignature)[(UINT)CompactBufferRootSignature::StreamCompactionUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 5, D3D12_SHADER_VISIBILITY_ALL, 1);


	voxelComputeRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

	return voxelComputeRootSignature;
}

void CVGI::PrefixSumVoxels::BuildPipelineState()
{
	std::shared_ptr<RootSignature> rootSig = BuildRootSignature();

	auto computeShader = CD3DX12_SHADER_BYTECODE((void*)g_pPrefixSum_CS, ARRAYSIZE(g_pPrefixSum_CS));

	std::unique_ptr<ComputePipelineState> voxelComputePSO = std::make_unique<ComputePipelineState>();
	voxelComputePSO->SetRootSignature(rootSig);
	voxelComputePSO->SetComputeShader(computeShader);
	voxelComputePSO->Finalize();
	voxelComputePSO->Name = Name;

	m_techniquePSO = std::move(voxelComputePSO);
}

const std::vector<UINT32>& CVGI::PrefixSumVoxels::GetIndirectionRankBuffer() const
{
	return m_indRnkBuffer;
}

const std::vector<UINT32>& CVGI::PrefixSumVoxels::GetIndirectionIndexBuffer() const
{
	return m_indIdxBuffer;
}

const std::vector<UINT32>& CVGI::PrefixSumVoxels::GetCompactedVoxelIndexBuffer() const
{
	return m_cmpIdxBuffer;
}

const std::vector<UINT32>& CVGI::PrefixSumVoxels::GetCompactedHashedBuffer() const
{
	return m_cmpHshBuffer;
}

const std::wstring CVGI::PrefixSumVoxels::Name = L"PrefixSumVoxels";
