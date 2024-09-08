#include "ClusterVisibility.h"
#include "DX12Lib/pch.h"
#include "WinPixEventRuntime/pix3.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;


void CVGI::ClusterVisibility::InitializeBuffers(UINT voxelCount, UINT clusterCount)
{
	// Faces buffer (u0)
	m_faceBufferManager.AddStructuredBuffer(1, sizeof(DirectX::XMUINT2));
	// Number of faces (u1);
	m_faceBufferManager.AddByteAddressBuffer();


	// Buffers for acceleration structures
	// Voxel AABB data (u0);
	m_aabbBufferManager.AddStructuredBuffer(voxelCount, sizeof(VoxelAABB));
	// Cluster Start Index and Count (u1);
	m_aabbBufferManager.AddStructuredBuffer(clusterCount, sizeof(AABBInfo));
	// Cluster AABB Counter (u2);
	m_aabbBufferManager.AddByteAddressBuffer();

	m_faceBufferManager.AllocateBuffers();
	m_aabbBufferManager.AllocateBuffers();

	m_numberOfVoxels = voxelCount;
	m_numberOfClusters = clusterCount;
}

void CVGI::ClusterVisibility::StartVisibility(ComputeContext& context, BufferManager& compactBufferManager)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 0), L"FaceCount");

	m_cbFaceCount.GridDimension = m_voxelTexDimensions;
	m_cbFaceCount.CurrentPhase = 0;
	m_cbFaceCount.VoxelCount = m_numberOfVoxels;

	compactBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	m_faceBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.FlushResourceBarriers();

	VisibilityPass(context, DirectX::XMUINT3(ceil(m_numberOfVoxels / 256), 1, 1), compactBufferManager);
	context.Flush();

	m_cbFaceCount.CurrentPhase = 1;

	m_numberOfFaces = *m_faceBufferManager.ReadFromBuffer<UINT32*>(context, 1);
	m_faceBufferManager.ResizeBuffer(0, m_numberOfFaces);
	m_faceBufferManager.ZeroBuffer(context, 1);

	VisibilityPass(context, DirectX::XMUINT3(ceil(m_numberOfFaces / 256), 1, 1), compactBufferManager);

	PIXEndEvent(context.m_commandList->Get());

	context.Flush();
}

void CVGI::ClusterVisibility::VisibilityPass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize, BufferManager& compactBufferManager)
{
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[FaceCountPsoName].get());

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)FaceCountRootSignature::FaceCountCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbFaceCount).GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)FaceCountRootSignature::CompactSRVTable, compactBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)FaceCountRootSignature::FaceCountUAVTable, m_faceBufferManager.GetUAVHandle());

	context.Dispatch(groupSize.x, groupSize.y, groupSize.z);
}

void CVGI::ClusterVisibility::StartAABBGeneration(ComputeContext& context, BufferManager& compactBufferManager, BufferManager& clusterBufferManager)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 0), L"AABBGeneration");

	compactBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	clusterBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	m_aabbBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.FlushResourceBarriers();

	m_cbAABBGeneration.GridDimension = m_voxelTexDimensions;
	m_cbAABBGeneration.ClusterCount = m_numberOfClusters;

	AABBGenerationPass(context, DirectX::XMUINT3(ceil(m_numberOfClusters / 256), 1, 1), compactBufferManager, clusterBufferManager);
	

	PIXEndEvent(context.m_commandList->Get());
	context.Flush();
}

void CVGI::ClusterVisibility::AABBGenerationPass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize, BufferManager& compactBufferManager, BufferManager& clusterBufferManager)
{
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[AABBGenerationPsoName].get());

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)AABBGenerationRootSignature::AABBGenerationCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbAABBGeneration).GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)AABBGenerationRootSignature::CompactSRVTable, compactBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)AABBGenerationRootSignature::ClusterSRVTable, clusterBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)AABBGenerationRootSignature::AABBGenerationUAVTable, m_aabbBufferManager.GetUAVHandle());

	context.Dispatch(groupSize.x, groupSize.y, groupSize.z);
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::ClusterVisibility::BuildFaceCountRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> faceCountRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)FaceCountRootSignature::Count, 1);
	faceCountRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*faceCountRootSignature)[(UINT)FaceCountRootSignature::FaceCountCBV].InitAsConstantBuffer(0);
	(*faceCountRootSignature)[(UINT)FaceCountRootSignature::CompactSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 0);
	(*faceCountRootSignature)[(UINT)FaceCountRootSignature::FaceCountUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 0);

	faceCountRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

	return faceCountRootSignature;
}


std::shared_ptr<DX12Lib::ComputePipelineState> CVGI::ClusterVisibility::BuildFaceCountPipelineState(std::shared_ptr<DX12Lib::RootSignature> rootSig)
{
	std::wstring shaderPath = Utils::ToWstring(SOURCE_DIR) + L"\\Shaders";
	std::wstring computeShaderPath = shaderPath + L"\\FaceCount_CS.hlsl";

	std::shared_ptr<Shader> computeShader = std::make_shared<Shader>(computeShaderPath, "CS", "cs_5_1");
	computeShader->Compile();

	std::shared_ptr<ComputePipelineState> faceCountComputePso = std::make_shared<ComputePipelineState>();
	faceCountComputePso->SetRootSignature(rootSig);
	faceCountComputePso->SetComputeShader(computeShader);
	faceCountComputePso->Finalize();
	faceCountComputePso->Name = FaceCountPsoName;

	return faceCountComputePso;
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::ClusterVisibility::BuildAABBGenerationRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> AABBComputeRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)AABBGenerationRootSignature::Count, 1);
	AABBComputeRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*AABBComputeRootSignature)[(UINT)AABBGenerationRootSignature::AABBGenerationCBV].InitAsConstantBuffer(0);
	(*AABBComputeRootSignature)[(UINT)AABBGenerationRootSignature::CompactSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 0);
	(*AABBComputeRootSignature)[(UINT)AABBGenerationRootSignature::ClusterSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 3, D3D12_SHADER_VISIBILITY_ALL, 1);
	(*AABBComputeRootSignature)[(UINT)AABBGenerationRootSignature::AABBGenerationUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 3, D3D12_SHADER_VISIBILITY_ALL, 0);

	AABBComputeRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

	return AABBComputeRootSignature;
}

std::shared_ptr<DX12Lib::ComputePipelineState> CVGI::ClusterVisibility::BuildAABBGenerationPipelineState(std::shared_ptr<DX12Lib::RootSignature> rootSig)
{
	std::wstring shaderPath = Utils::ToWstring(SOURCE_DIR) + L"\\Shaders";
	std::wstring computeShaderPath = shaderPath + L"\\ComputeAABB_CS.hlsl";

	std::shared_ptr<Shader> computeShader = std::make_shared<Shader>(computeShaderPath, "CS", "cs_5_1");
	computeShader->Compile();

	std::shared_ptr<ComputePipelineState> AABBGenerationComputePso = std::make_shared<ComputePipelineState>();
	AABBGenerationComputePso->SetRootSignature(rootSig);
	AABBGenerationComputePso->SetComputeShader(computeShader);
	AABBGenerationComputePso->Finalize();
	AABBGenerationComputePso->Name = AABBGenerationPsoName;

	return AABBGenerationComputePso;
}

void CVGI::ClusterVisibility::BuildAccelerationStructures(ComputeContext& context)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 0), L"BuildAccelerationStructures");

	std::vector<AABBInfo> clusterAABBInfo(m_numberOfClusters);

	memcpy(clusterAABBInfo.data(), m_aabbBufferManager.ReadFromBuffer<AABBInfo*>(context, 1), m_numberOfClusters * sizeof(AABBInfo));
	
	std::erase_if(clusterAABBInfo, [](const AABBInfo& element) {
		return element.ClusterElementCount == 0;
		});

	std::vector<AABB> clusterAABBs(clusterAABBInfo.size());

	for (size_t i = 0; i < clusterAABBInfo.size(); i++)
	{
		assert(clusterAABBInfo[i].ClusterElementCount > 0 && "Error building BLAS with no geometry");
		clusterAABBs[i].min = clusterAABBInfo[i].ClusterMin;
		clusterAABBs[i].max = clusterAABBInfo[i].ClusterMax;
	}

	Octree octree(clusterAABBs, DirectX::XMFLOAT3(m_voxelTexDimensions.x, m_voxelTexDimensions.y, m_voxelTexDimensions.z));

	UINT maxSize = 200;

	octree.CreateOctree(20, maxSize);

	auto leavesGroups = octree.GetLeaves();

	GPUBuffer& aabbBuffer = m_aabbBufferManager.GetBuffer(0);
	aabbBuffer.GetComPtr()->SetName(L"ClusterVisibility::aabbBuffer");
	context.TransitionResource(aabbBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, true);
	
	auto baseAddress = aabbBuffer.GetGpuVirtualAddress();


	for (std::vector<unsigned int>& leaf : leavesGroups)
	{
		std::shared_ptr<BottomLevelAccelerationStructure> blas = std::make_shared<BottomLevelAccelerationStructure>();

		for (unsigned int index : leaf)
		{
			auto& clusterInfo = clusterAABBInfo[index];
			blas->AddGeometry(clusterInfo.ClusterElementCount, 
				baseAddress + clusterInfo.ClusterStartIndex * sizeof(VoxelAABB),
				sizeof(VoxelAABB));
		}

		blas->Build(context);
		m_TLAS.AddBLAS(blas);
	}

	m_TLAS.Build(context);

	PIXEndEvent(context.m_commandList->Get());
	context.Flush();
}
