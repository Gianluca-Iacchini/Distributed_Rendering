#include "VoxelBufferManager.h"
#include "DX12Lib/pch.h"
#include "CVGIDataTypes.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"
#include "WinPixEventRuntime/pix3.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;



CVGI::VoxelBufferManager::VoxelBufferManager()
{
	m_buffers[(UINT)BufferType::FragmentCounter] = &m_fragmentCounterBuffer;
	m_buffers[(UINT)BufferType::VoxelCounter] = &m_voxelCounterBuffer;
	m_buffers[(UINT)BufferType::VoxelOccupied] = &m_voxelOccupiedBuffer;
	m_buffers[(UINT)BufferType::VoxelIndex] = &m_voxelIndexBuffer;
	m_buffers[(UINT)BufferType::FragmentData] = &m_fragmentDataBuffer;
	m_buffers[(UINT)BufferType::NextIndex] = &m_nextIndexBuffer;
	m_buffers[(UINT)BufferType::HashedBuffer] = &m_hashedBuffer;
	m_buffers[(UINT)BufferType::PrefixSum] = &m_prefixSumBuffer;
	m_buffers[(UINT)BufferType::IndirectionRankBuffer] = &m_indirectionRankBuffer;
	m_buffers[(UINT)BufferType::IndirectionIndexBuffer] = &m_indirectionIndexBuffer;
	m_buffers[(UINT)BufferType::CompactedVoxelIndex] = &m_compactedVoxelIndexBuffer;
	m_buffers[(UINT)BufferType::CompactedHashedBuffer] = &m_compactedHashedBuffer;
	m_buffers[(UINT)BufferType::ClusterData] = &m_clusterDataBuffer;

}

void CVGI::VoxelBufferManager::SetupFirstVoxelPassBuffers(DirectX::XMFLOAT3 voxelTexDimension)
{
	SetVoxelTextureDimension(voxelTexDimension);

	m_voxelizationUAVStart = Renderer::s_textureHeap->Alloc(7);
	m_streamCompactUAVStart = Renderer::s_textureHeap->Alloc(5);
	m_clusterizeUAVStart = Renderer::s_textureHeap->Alloc(7);

	m_voxelizationSRVStart = Renderer::s_textureHeap->Alloc(2);
	m_streamCompactSRVStart = Renderer::s_textureHeap->Alloc(4);
	m_clusterizeSRVStart = Renderer::s_textureHeap->Alloc(3);

	// Using bit representation of voxels to store whether a voxel is occupied or not.
	// Each bit represents wheter a specific voxel is occupied or not. We can store 32 voxels in a single UINT32,
	// So we divide the total number of voxels by 32 to get the size of the buffer.
	UINT32 voxelOccupiedSize = (m_voxelLinearSize + 31) / 32;

	m_fragmentCounterBuffer.Create(1, sizeof(UINT32));
	m_voxelCounterBuffer.Create(1, sizeof(UINT32));
	m_voxelOccupiedBuffer.Create(voxelOccupiedSize, sizeof(UINT32));
	m_voxelIndexBuffer.Create(m_voxelLinearSize, sizeof(UINT32));


	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferUAVStart(BufferType::FragmentCounter), m_fragmentCounterBuffer.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferUAVStart(BufferType::VoxelCounter) , m_voxelCounterBuffer.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferUAVStart(BufferType::VoxelOccupied) , m_voxelOccupiedBuffer.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferUAVStart(BufferType::VoxelIndex) , m_voxelIndexBuffer.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	m_cbCompactBuffer.VoxelTextureDimensions= DirectX::XMUINT3(UINT(voxelTexDimension.x), UINT(voxelTexDimension.y), UINT(voxelTexDimension.z));
}

void CVGI::VoxelBufferManager::SetupSecondVoxelPassBuffers(DX12Lib::CommandContext& context, UINT numFragments)
{
	m_fragmentDataBuffer.Create(numFragments, sizeof(FragmentData));
	m_nextIndexBuffer.Create(numFragments, sizeof(UINT32));
	m_hashedBuffer.Create(numFragments, sizeof(UINT32));
	m_indirectionRankBuffer.Create(m_voxelTexDimension.y * m_voxelTexDimension.z, sizeof(UINT32));
	m_indirectionIndexBuffer.Create(m_voxelTexDimension.y * m_voxelTexDimension.z, sizeof(UINT32));

	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferUAVStart(BufferType::FragmentData), m_fragmentDataBuffer.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferUAVStart(BufferType::NextIndex), m_nextIndexBuffer.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferUAVStart(BufferType::HashedBuffer), m_hashedBuffer.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferUAVStart(BufferType::IndirectionRankBuffer), m_indirectionRankBuffer.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferUAVStart(BufferType::IndirectionIndexBuffer), m_indirectionIndexBuffer.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferSRVStart(BufferType::FragmentData), m_fragmentDataBuffer.GetSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferSRVStart(BufferType::NextIndex), m_nextIndexBuffer.GetSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferSRVStart(BufferType::IndirectionRankBuffer), m_indirectionRankBuffer.GetSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferSRVStart(BufferType::IndirectionIndexBuffer), m_indirectionIndexBuffer.GetSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);


	// Since voxelCounterBuffer is still unused and is the same size and type as fragmentCounterBuffer, we can use it to
	// reset the fragment counter buffer.
	context.CopyBuffer(m_fragmentCounterBuffer, m_voxelCounterBuffer);

	UploadBuffer voxelIndexUploader;
	voxelIndexUploader.Create(m_voxelLinearSize * sizeof(UINT32));

	void* mappedData = voxelIndexUploader.Map();

	for (UINT32 i = 0; i < m_voxelLinearSize; i++)
	{
		((UINT32*)mappedData)[i] = UINT32_MAX;
	}

	// Not using CommandContext.CopyBuffer because upload buffer should not be transitioned from the GENERIC_READ state
	context.TransitionResource(m_voxelIndexBuffer, D3D12_RESOURCE_STATE_COPY_DEST, true);
	context.m_commandList->Get()->CopyResource(m_voxelIndexBuffer.Get(), voxelIndexUploader.Get());

	context.m_commandList->Get()->CopyBufferRegion(
		m_indirectionIndexBuffer.Get(),
		0,
		voxelIndexUploader.Get(),
		0,
		sizeof(UINT32) * m_voxelTexDimension.y * m_voxelTexDimension.z);

	//context.m_commandList->Get()->CopyBufferRegion(
	//	m_indirectionRankBuffer.Get(),
	//	0,
	//	voxelIndexUploader.Get(),
	//	0,
	//	sizeof(UINT32) * m_voxelTexDimension.x * m_voxelTexDimension.y);

	// Technically flushing right now is not efficient, since the next call after this should be the second voxel draw pass,
	// however this function should ideally be only called once at the start and doing it this way we allows us to unmap the upload
	// buffer right away.
	context.Flush(true);

	voxelIndexUploader.Unmap();
}

void CVGI::VoxelBufferManager::SetupCompactBuffers()
{
	v_prefixBufferSizeForStep.resize(5); // Up to four levels needed to the prefix parallel sum
	memset(v_prefixBufferSizeForStep.data(), 0, v_prefixBufferSizeForStep.size() * size_t(sizeof(UINT32)));

	v_prefixBufferSizeForStep[0] = m_voxelLinearSize / m_elementsPerThread;

	m_prefixBufferSize = 0;
	// The maximum number of elements in bufferHistogramBuffer is 536862720,
	// given that each thread processes 128 elements of bufferHistogramBuffer in the initial pass, up to
	// four levels are needed to complete the algorithm.
	m_prefixBufferSize += UINT32(v_prefixBufferSizeForStep[0]);
	float prefixSumNumElemenCurrentStep = float(v_prefixBufferSizeForStep[0]);
	bool stop = ((prefixSumNumElemenCurrentStep / (float)m_elementsPerThread) <= 1.0f);

	m_reduceStepCount = 1;



	while (!stop)
	{
		prefixSumNumElemenCurrentStep = ceilf(prefixSumNumElemenCurrentStep / m_elementsPerThread);
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

	m_prefixSumBuffer.Create(m_prefixBufferSize, sizeof(UINT32));
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferUAVStart(BufferType::PrefixSum), m_prefixSumBuffer.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	


	DXLIB_INFO("Number of steps reduce: {0}", m_reduceStepCount);
	DXLIB_INFO("Planar buffer size: {0}", m_prefixBufferSize);
}

void CVGI::VoxelBufferManager::CompactBuffers()
{
	ComputeContext& context = ComputeContext::Begin();

	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 0), L"PrefixSumPass");


	m_cbCompactBuffer.ElementsPerThread = m_elementsPerThread;

	// Reduce
	for (m_currentStep = 0; m_currentStep < m_reduceStepCount; m_currentStep++)
	{
		CompactBufferPass(context, v_prefixBufferSizeForStep[m_currentStep]);
		context.Flush();
	}

	UINT32 prefixSumValue = 0;

	{
		UINT32* prefixBufferData = this->ReadFromBuffer<UINT32*>(context, BufferType::PrefixSum);
		prefixSumValue = prefixBufferData[m_prefixBufferSize - 1];
	}
	DXLIB_CORE_INFO("Prefix sum value after reduce: {0}", prefixSumValue);

	m_compactedVoxelIndexBuffer.Create(prefixSumValue, sizeof(UINT32));
	m_compactedHashedBuffer.Create(prefixSumValue, sizeof(UINT32));

	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferUAVStart(BufferType::CompactedVoxelIndex), m_compactedVoxelIndexBuffer.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferUAVStart(BufferType::CompactedHashedBuffer), m_compactedHashedBuffer.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferSRVStart(BufferType::CompactedVoxelIndex), m_compactedVoxelIndexBuffer.GetSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferSRVStart(BufferType::CompactedHashedBuffer), m_compactedHashedBuffer.GetSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);


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
		CompactBufferPass(context, UINT32(ceilf(float(v_prefixBufferSizeForStep[m_currentStep]) / float(m_elementsPerThread))));
		m_currentStep--;
		context.Flush();
	}
	
	PIXSetMarker(context.m_commandList->Get(), PIX_COLOR(0, 0, 128), L"Copy Pass");
	m_currentPhase = 2;

	{
		auto val = this->ReadFromBuffer<UINT32*>(context, BufferType::PrefixSum);
		prefixSumValue = val[m_prefixBufferSize - 1];
	}
	DXLIB_CORE_INFO("Prefix sum value after sweep: {0}", prefixSumValue);
	
	CompactBufferPass(context, v_prefixBufferSizeForStep[0]);

	{
		auto val = this->ReadFromBuffer<UINT32*>(context, BufferType::PrefixSum);
		prefixSumValue = val[m_prefixBufferSize - 1];
	}
	DXLIB_CORE_INFO("Prefix sum value after copy: {0}", prefixSumValue);

	PIXEndEvent(context.m_commandList->Get());
	context.Finish();
}

void CVGI::VoxelBufferManager::ClusterizeBuffers()
{
	ComputeContext& context = ComputeContext::Begin();

	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 0), L"ClusterizePass");

	m_cbClusterizeBuffer.CurrentPhase = 0;
	m_cbClusterizeBuffer.VoxelCount = m_voxelCount;
	m_cbClusterizeBuffer.m = 1;
	m_cbClusterizeBuffer.K = m_numberOfClusters;

	m_cbClusterizeBuffer.VoxelTextureDimensions = DirectX::XMUINT3(UINT(m_voxelTexDimension.x), UINT(m_voxelTexDimension.y), UINT(m_voxelTexDimension.z));
	m_cbClusterizeBuffer.S = UINT32(ceil(m_superPixelArea));

	m_cbClusterizeBuffer.TileGridDimension = TileGridDimension;

	context.TransitionResource(m_fragmentDataBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	context.TransitionResource(m_nextIndexBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

	context.TransitionResource(m_indirectionRankBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	context.TransitionResource(m_indirectionIndexBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	context.TransitionResource(m_compactedVoxelIndexBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	context.TransitionResource(m_compactedHashedBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

	context.TransitionResource(m_clusterDataBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.TransitionResource(m_voxelClusterLinkedList, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.TransitionResource(m_assignemtMapBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.TransitionResource(m_distanceMapBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.TransitionResource(m_tileTexture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.TransitionResource(m_nextClusterList, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.TransitionResource(m_clusterCounterBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);


	context.FlushResourceBarriers();

	ClusterizeBufferPass(context);

	PIXEndEvent(context.m_commandList->Get());

	context.Finish();
}

void CVGI::VoxelBufferManager::InitializeClusters()
{
	m_fragmentCounterBuffer.OnDestroy();
	m_voxelCounterBuffer.OnDestroy();
	m_voxelOccupiedBuffer.OnDestroy();
	m_voxelIndexBuffer.OnDestroy();
	m_hashedBuffer.OnDestroy();
	m_prefixSumBuffer.OnDestroy();

	m_numberOfClusters = MathHelper::Min(10000u, (UINT32)(m_voxelCount * 0.1f));

	m_superPixelArea = cbrtf(m_voxelLinearSize / m_numberOfClusters);
	float denominator = 2.0f * m_superPixelArea;

	TileGridDimension = { (UINT)ceilf(m_voxelTexDimension.x / denominator), (UINT)ceilf(m_voxelTexDimension.y / denominator), (UINT)ceilf(m_voxelTexDimension.z / denominator) };
	
	UINT32 clusterTileOccupancy = (m_numberOfClusters + 31) / 32;

	m_clusterDataBuffer.Create(m_numberOfClusters, sizeof(ClusterData));
	m_assignemtMapBuffer.Create(m_voxelCount, sizeof(UINT32));
	m_voxelClusterLinkedList.Create(m_voxelCount, sizeof(UINT32));
	m_distanceMapBuffer.Create(m_voxelCount, sizeof(float));
	m_nextClusterList.Create(m_numberOfClusters, sizeof(UINT16));
	m_tileTexture.Create3D(TileGridDimension.x, TileGridDimension.y, TileGridDimension.z, 1, DXGI_FORMAT_R16_UINT);
	m_clusterCounterBuffer.Create(1, sizeof(UINT32));

	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferUAVStart(BufferType::ClusterData), m_clusterDataBuffer.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferUAVStart(BufferType::AssignmentMap), m_assignemtMapBuffer.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferUAVStart(BufferType::NextVoxel), m_voxelClusterLinkedList.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferUAVStart(BufferType::DistanceMap), m_distanceMapBuffer.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferUAVStart(BufferType::NextCluster), m_nextClusterList.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferUAVStart(BufferType::TileBuffer), m_tileTexture.GetUAV(0), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferUAVStart(BufferType::ClusterCounterBuffer), m_clusterCounterBuffer.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, GetBufferSRVStart(BufferType::ClusterData), m_clusterDataBuffer.GetSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

void CVGI::VoxelBufferManager::CompactBufferPass(DX12Lib::ComputeContext& context, UINT numGroupsX)
{
	m_cbCompactBuffer.CurrentStep = m_currentStep;
	m_cbCompactBuffer.CurrentPhase = m_currentPhase;

	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[CompactBufferPsoName].get());

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)CompactBufferRootSignature::PrefixSumCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbCompactBuffer).GpuAddress());

	context.TransitionResource(m_voxelIndexBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.TransitionResource(m_prefixSumBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);

	

	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)CompactBufferRootSignature::VoxelizeUAVTable, m_voxelizationUAVStart);
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)CompactBufferRootSignature::StreamCompactionUAVTable, m_streamCompactUAVStart);

	// Align to 127 because we have 127 threads per group
	UINT groupSize = (numGroupsX + 127) / 128;
	context.Dispatch(numGroupsX, 1 , 1);
}

void CVGI::VoxelBufferManager::ClusterizeBufferPass(DX12Lib::ComputeContext& context)
{
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[ClusterizeBufferPsoName].get());

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)ClusterizeRootSignature::ClusterizeCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbClusterizeBuffer).GpuAddress());
	
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)ClusterizeRootSignature::VoxelBuffersSRVTable, m_voxelizationSRVStart);
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)ClusterizeRootSignature::StreamCompactionSRVTable, m_streamCompactSRVStart);
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)ClusterizeRootSignature::ClusterizeUAVTable, m_clusterizeUAVStart);

	context.Dispatch(TileGridDimension.x, TileGridDimension.y, TileGridDimension.z);
}

DX12Lib::DescriptorHandle& CVGI::VoxelBufferManager::GetBufferSRVStart(BufferType type)
{
	switch (type)
	{
	case CVGI::BufferType::FragmentData:
		return m_voxelizationSRVStart + Graphics::Renderer::s_textureHeap->GetDescriptorSize() * 0;
	case CVGI::BufferType::NextIndex:
		return m_voxelizationSRVStart + Graphics::Renderer::s_textureHeap->GetDescriptorSize() * 1;

	case CVGI::BufferType::IndirectionRankBuffer:
		return m_streamCompactSRVStart + Graphics::Renderer::s_textureHeap->GetDescriptorSize() * 0;
	case CVGI::BufferType::IndirectionIndexBuffer:
		return m_streamCompactSRVStart + Graphics::Renderer::s_textureHeap->GetDescriptorSize() * 1;
	case CVGI::BufferType::CompactedVoxelIndex:
		return m_streamCompactSRVStart + Graphics::Renderer::s_textureHeap->GetDescriptorSize() * 2;
	case CVGI::BufferType::CompactedHashedBuffer:
		return m_streamCompactSRVStart + Graphics::Renderer::s_textureHeap->GetDescriptorSize() * 3;

	case CVGI::BufferType::ClusterData:
		return m_clusterizeSRVStart + Graphics::Renderer::s_textureHeap->GetDescriptorSize() * 0;
	case CVGI::BufferType::NextVoxel:
		return m_clusterizeSRVStart + Graphics::Renderer::s_textureHeap->GetDescriptorSize() * 1;
	case CVGI::BufferType::AssignmentMap:
		return m_clusterizeSRVStart + Graphics::Renderer::s_textureHeap->GetDescriptorSize() * 2;


	default:
		throw std::exception("Invalid buffer type");
	}
}

DX12Lib::DescriptorHandle& CVGI::VoxelBufferManager::GetBufferUAVStart(BufferType type)
{
	UINT typeIndex = (UINT)type;

	if (typeIndex >= (UINT)BufferType::ClusterData)
	{
		UINT offset = typeIndex - (UINT)BufferType::ClusterData;

		return m_clusterizeUAVStart + Graphics::Renderer::s_textureHeap->GetDescriptorSize() * offset;
	}
	else if (typeIndex >= (UINT)BufferType::IndirectionRankBuffer)
	{
		UINT offset = typeIndex - (UINT)BufferType::IndirectionRankBuffer;

		return m_streamCompactUAVStart + Graphics::Renderer::s_textureHeap->GetDescriptorSize() * offset;
	}
	else
	{
		return m_voxelizationUAVStart + Graphics::Renderer::s_textureHeap->GetDescriptorSize() * typeIndex;
	}
}


std::shared_ptr<DX12Lib::RootSignature> CVGI::VoxelBufferManager::BuildCompactBufferRootSignature()
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

std::shared_ptr<DX12Lib::ComputePipelineState> CVGI::VoxelBufferManager::BuildCompactBufferPso(std::shared_ptr<DX12Lib::RootSignature> voxelRootSig)
{
	std::wstring shaderPath = Utils::ToWstring(SOURCE_DIR) + L"\\Shaders";
	std::wstring computeShaderPath = shaderPath + L"\\PrefixSum_CS.hlsl";

	std::shared_ptr<Shader> computeShader = std::make_shared<Shader>(computeShaderPath, "CS", "cs_5_1");
	computeShader->Compile();

	std::shared_ptr<ComputePipelineState> voxelComputePSO = std::make_shared<ComputePipelineState>();
	voxelComputePSO->SetRootSignature(voxelRootSig);
	voxelComputePSO->SetComputeShader(computeShader);
	voxelComputePSO->Finalize();
	voxelComputePSO->Name = VoxelBufferManager::CompactBufferPsoName;

	return voxelComputePSO;
}



std::shared_ptr<DX12Lib::RootSignature> CVGI::VoxelBufferManager::BuildClusterizeRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> clusterRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)ClusterizeRootSignature::Count, 1);
	clusterRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*clusterRootSignature)[(UINT)ClusterizeRootSignature::ClusterizeCBV].InitAsConstantBuffer(0);
	(*clusterRootSignature)[(UINT)ClusterizeRootSignature::VoxelBuffersSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 0);
	(*clusterRootSignature)[(UINT)ClusterizeRootSignature::StreamCompactionSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 1);
	(*clusterRootSignature)[(UINT)ClusterizeRootSignature::ClusterizeUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 7, D3D12_SHADER_VISIBILITY_ALL, 0);

	clusterRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

	return clusterRootSignature;
}

std::shared_ptr<DX12Lib::ComputePipelineState> CVGI::VoxelBufferManager::BuildClulsterizePso(std::shared_ptr<DX12Lib::RootSignature> voxelRootSig)
{
	std::wstring shaderPath = Utils::ToWstring(SOURCE_DIR) + L"\\Shaders";
	std::wstring computeShaderPath = shaderPath + L"\\FastSlic_CS.hlsl";

	std::shared_ptr<Shader> computeShader = std::make_shared<Shader>(computeShaderPath, "CS", "cs_5_1");
	computeShader->Compile();

	std::shared_ptr<ComputePipelineState> voxelClusterizeComputePso = std::make_shared<ComputePipelineState>();
	voxelClusterizeComputePso->SetRootSignature(voxelRootSig);
	voxelClusterizeComputePso->SetComputeShader(computeShader);
	voxelClusterizeComputePso->Finalize();
	voxelClusterizeComputePso->Name = VoxelBufferManager::ClusterizeBufferPsoName;

	return voxelClusterizeComputePso;
}

const std::wstring CVGI::VoxelBufferManager::CompactBufferPsoName = L"PSO_PREFIX_SUM";
const std::wstring CVGI::VoxelBufferManager::ClusterizeBufferPsoName = L"PSO_FAST_SLIC ";