#include "PrefixSumVoxels.h"
#include "DX12Lib/pch.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"
#include "WinPixEventRuntime/pix3.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;

void CVGI::PrefixSumVoxels::InitializeBuffers(DX12Lib::CommandContext& context)
{
	// Indirection Rank Buffer (u0)
	m_bufferManager.AddStructuredBuffer(m_voxelizationSize.y * m_voxelizationSize.z, sizeof(UINT32));

	// Indirection Index Buffer (u1)
	UINT indIndBuffIdx = m_bufferManager.AddStructuredBuffer(m_voxelizationSize.y * m_voxelizationSize.z, sizeof(UINT32));

	UploadBuffer uploadBuffer;
	uploadBuffer.Create(m_voxelizationSize.y * m_voxelizationSize.z * sizeof(UINT32));

	void* mappedData = uploadBuffer.Map();

	for (UINT32 i = 0; i < m_voxelizationSize.y * m_voxelizationSize.z; i++)
	{
		((UINT32*)mappedData)[i] = UINT_MAX;
	}


	DX12Lib::GPUBuffer& indIndBuff = m_bufferManager.GetBuffer(indIndBuffIdx);
	// Not using CommandContext.CopyBuffer because upload buffer should not be transitioned from the GENERIC_READ state
	context.TransitionResource(indIndBuff, D3D12_RESOURCE_STATE_COPY_DEST, true);
	context.m_commandList->Get()->CopyResource(indIndBuff.Get(), uploadBuffer.Get());

	ComputePrefixSumVariables();

	// Voxel Index Compacted Buffer (u2)
	// Will be resized later
	m_bufferManager.AddStructuredBuffer(1, sizeof(UINT32));

	// Compact hash buffer (u3)
	// Will be resized later
	m_bufferManager.AddStructuredBuffer(1, sizeof(UINT32));

	// Prefix Sum Buffer (u4)
	m_bufferManager.AddStructuredBuffer(m_prefixBufferSize, sizeof(UINT32));

	m_bufferManager.AllocateBuffers();

	DXLIB_INFO("Number of steps reduce: {0}", m_reduceStepCount);
	DXLIB_INFO("Planar buffer size: {0}", m_prefixBufferSize);

	context.Flush(true);

	uploadBuffer.Unmap();
}



void CVGI::PrefixSumVoxels::ComputePrefixSumVariables()
{
	v_prefixBufferSizeForStep.resize(5); // Up to four levels needed to the prefix parallel sum
	memset(v_prefixBufferSizeForStep.data(), 0, v_prefixBufferSizeForStep.size() * size_t(sizeof(UINT32)));

	v_prefixBufferSizeForStep[0] = (m_voxelizationSize.x * m_voxelizationSize.y * m_voxelizationSize.z) / ELEMENTS_PER_THREAD;

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

void CVGI::PrefixSumVoxels::StartPrefixSum(BufferManager* voxelBufferManager)
{

	ComputeContext& context = ComputeContext::Begin();

	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 0), L"PrefixSumPass");


	m_cbCompactBuffer.ElementsPerThread = ELEMENTS_PER_THREAD;

	// Reduce
	for (m_currentStep = 0; m_currentStep < m_reduceStepCount; m_currentStep++)
	{
		CompactBufferPass(context, v_prefixBufferSizeForStep[m_currentStep], voxelBufferManager);
		context.Flush();
	}

	UINT32 prefixSumValue = 0;
	{
		UINT32* prefixBufferData = m_bufferManager.ReadFromBuffer<UINT32*>(context, (UINT)PrefixSumBufferType::PrefixSum);
		prefixSumValue = prefixBufferData[m_prefixBufferSize - 1];
	}
	DXLIB_CORE_INFO("Prefix sum value after reduce: {0}", prefixSumValue);

	m_bufferManager.ResizeBuffer((UINT)PrefixSumBufferType::CompactedVoxelIndex, prefixSumValue);
	m_bufferManager.ResizeBuffer((UINT)PrefixSumBufferType::CompactedHashedBuffer, prefixSumValue);


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
		CompactBufferPass(context, UINT32(ceilf(float(v_prefixBufferSizeForStep[m_currentStep]) / float(ELEMENTS_PER_THREAD))), voxelBufferManager);
		m_currentStep--;
		context.Flush();
	}

	PIXSetMarker(context.m_commandList->Get(), PIX_COLOR(0, 0, 128), L"Copy Pass");
	m_currentPhase = 2;

	{
		auto val = m_bufferManager.ReadFromBuffer<UINT32*>(context, (UINT)PrefixSumBufferType::PrefixSum);
		prefixSumValue = val[m_prefixBufferSize - 1];
	}

	DXLIB_CORE_INFO("Prefix sum value after sweep: {0}", prefixSumValue);

	CompactBufferPass(context, v_prefixBufferSizeForStep[0], voxelBufferManager);

	{
		auto val = m_bufferManager.ReadFromBuffer<UINT32*>(context, (UINT)PrefixSumBufferType::PrefixSum);
		prefixSumValue = val[m_prefixBufferSize - 1];
	}
	DXLIB_CORE_INFO("Prefix sum value after copy: {0}", prefixSumValue);

	PIXEndEvent(context.m_commandList->Get());
	context.Finish();
}

void CVGI::PrefixSumVoxels::CompactBufferPass(DX12Lib::ComputeContext& context, UINT32 numGroupsX, BufferManager* voxelBufferManager)
{
	m_cbCompactBuffer.CurrentStep = m_currentStep;
	m_cbCompactBuffer.CurrentPhase = m_currentPhase;

	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[PrefixSumPsoName].get());

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)CompactBufferRootSignature::PrefixSumCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbCompactBuffer).GpuAddress());

	m_bufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);


	if (voxelBufferManager != nullptr)
	{
		voxelBufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)CompactBufferRootSignature::VoxelizeUAVTable, voxelBufferManager->GetUAVHandle());
	}
	
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)CompactBufferRootSignature::StreamCompactionUAVTable, m_bufferManager.GetUAVHandle());

	// Align to 127 because we have 127 threads per group
	UINT groupSize = (numGroupsX + 127) / 128;
	context.Dispatch(numGroupsX, 1, 1);
}

void CVGI::PrefixSumVoxels::DeleteTemporaryBuffers()
{
	m_bufferManager.RemoveBuffer((UINT)PrefixSumBufferType::PrefixSum);
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::PrefixSumVoxels::BuildPrefixSumRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> voxelComputeRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)CompactBufferRootSignature::Count, 1);
	voxelComputeRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*voxelComputeRootSignature)[(UINT)CompactBufferRootSignature::PrefixSumCBV].InitAsConstantBuffer(0);
	(*voxelComputeRootSignature)[(UINT)CompactBufferRootSignature::VoxelizeUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 3);
	(*voxelComputeRootSignature)[(UINT)CompactBufferRootSignature::StreamCompactionUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 5, D3D12_SHADER_VISIBILITY_ALL, 1);


	voxelComputeRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

	return voxelComputeRootSignature;
}

std::shared_ptr<DX12Lib::ComputePipelineState> CVGI::PrefixSumVoxels::BuildPrefixSumPipelineState(std::shared_ptr<DX12Lib::RootSignature> rootSig)
{
	std::wstring shaderPath = Utils::ToWstring(SOURCE_DIR) + L"\\Shaders";
	std::wstring computeShaderPath = shaderPath + L"\\PrefixSum_CS.hlsl";

	std::shared_ptr<Shader> computeShader = std::make_shared<Shader>(computeShaderPath, "CS", "cs_5_1");
	computeShader->Compile();

	std::shared_ptr<ComputePipelineState> voxelComputePSO = std::make_shared<ComputePipelineState>();
	voxelComputePSO->SetRootSignature(rootSig);
	voxelComputePSO->SetComputeShader(computeShader);
	voxelComputePSO->Finalize();
	voxelComputePSO->Name = PrefixSumPsoName;

	return voxelComputePSO;
}
