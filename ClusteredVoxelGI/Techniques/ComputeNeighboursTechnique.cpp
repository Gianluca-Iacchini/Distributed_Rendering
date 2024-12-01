#include "ComputeNeighboursTechnique.h"
#include "DX12Lib/DXWrapper/RootSignature.h"
#include "DX12Lib/pch.h"
#include "ClusterVoxels.h"
#include "../Data/Shaders/ComputeNeighbour_CS.h"

using namespace DX12Lib;
using namespace Graphics;
using namespace CVGI;

CVGI::ComputeNeighboursTechnique::ComputeNeighboursTechnique(std::shared_ptr<TechniqueData> data)
{
	m_bufferManager = std::make_shared<BufferManager>();
	data->SetBufferManager(Name, m_bufferManager);
	m_data = data;
}

void CVGI::ComputeNeighboursTechnique::InitializeBuffers()
{
	

	m_bufferManager->AddStructuredBuffer(m_data->GetClusterCount(), sizeof(UINT32));
	m_bufferManager->AddByteAddressBuffer(2);

	m_bufferManager->AllocateBuffers();
}

void CVGI::ComputeNeighboursTechnique::PerformTechnique(DX12Lib::ComputeContext& context)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR_DEFAULT, Name.c_str());

	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[Name].get());

	UINT32 clusterCount = m_data->GetClusterCount();
	UINT32 totalComputation = clusterCount * (clusterCount + 1) / 2;

	UINT elementsPerThread = 50;

	m_cbNeighbour.ElementsPerThread = elementsPerThread;
	m_cbNeighbour.ClusterCount = clusterCount;
	m_cbNeighbour.TotalComputationCount = totalComputation;

	TechniquePass(context, DirectX::XMUINT3((UINT)ceilf(totalComputation / (256.0f * elementsPerThread)), 1, 1));

	PIXEndEvent(context.m_commandList->Get());
}

void CVGI::ComputeNeighboursTechnique::TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize)
{
	context.SetPipelineState(Renderer::s_PSOs[Name].get());
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());

	auto& clusterVoxelBufferManager = m_data->GetBufferManager(ClusterVoxels::Name);

	clusterVoxelBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.AddUAVIfNoBarriers(m_bufferManager->GetBuffer(0));
	context.FlushResourceBarriers();

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)ComputeNeighbourRootParameter::VoxelCommonsCBV, m_data->GetVoxelCommonsResource().GpuAddress());
	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)ComputeNeighbourRootParameter::NeighbourCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbNeighbour).GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)ComputeNeighbourRootParameter::ClusterVoxelsUAVTable, clusterVoxelBufferManager.GetUAVHandle());

	context.Dispatch(groupSize.x, groupSize.y, groupSize.z);
}

std::shared_ptr<DX12Lib::PipelineState> CVGI::ComputeNeighboursTechnique::BuildPipelineState()
{
	std::shared_ptr<DX12Lib::RootSignature> computeNeighborRootSig = BuildRootSignature();

	auto shaderBlob = CD3DX12_SHADER_BYTECODE((void*)g_pComputeNeighbour_CS, ARRAYSIZE(g_pComputeNeighbour_CS));

	std::shared_ptr<DX12Lib::ComputePipelineState> computeNeighborPso = std::make_shared<DX12Lib::ComputePipelineState>();
	computeNeighborPso->SetRootSignature(computeNeighborRootSig);
	computeNeighborPso->SetComputeShader(shaderBlob);
	computeNeighborPso->Finalize();
	computeNeighborPso->Name = Name;

	return computeNeighborPso;
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::ComputeNeighboursTechnique::BuildRootSignature()
{
	std::shared_ptr<DX12Lib::RootSignature> computeNeighborRootSig = std::make_shared<DX12Lib::RootSignature>((UINT)ComputeNeighbourRootParameter::Count, 0);
	(*computeNeighborRootSig)[(UINT)ComputeNeighbourRootParameter::VoxelCommonsCBV].InitAsConstantBuffer(0);
	(*computeNeighborRootSig)[(UINT)ComputeNeighbourRootParameter::NeighbourCBV].InitAsConstantBuffer(1);
	(*computeNeighborRootSig)[(UINT)ComputeNeighbourRootParameter::ClusterVoxelsUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 1, D3D12_SHADER_VISIBILITY_ALL, 0);

	computeNeighborRootSig->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

	return computeNeighborRootSig;
}

const std::wstring CVGI::ComputeNeighboursTechnique::Name = L"ComputeNeighboursTechnique";
