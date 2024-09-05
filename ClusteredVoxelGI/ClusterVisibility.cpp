#include "ClusterVisibility.h"
#include "DX12Lib/pch.h"
#include "WinPixEventRuntime/pix3.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;


void CVGI::ClusterVisibility::InitializeBuffers(UINT voxelCount)
{
	// Faces buffer (u0)
	m_bufferManager.AddStructuredBuffer(1, sizeof(DirectX::XMUINT2));
	// Number of faces (u1);
	m_bufferManager.AddByteAddressBuffer();

	m_bufferManager.AllocateBuffers();

	m_numberOfVoxels = voxelCount;
}

void CVGI::ClusterVisibility::StartVisibility(BufferManager& compactBufferManager)
{
	ComputeContext& context = ComputeContext::Begin();

	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 0), L"FaceCount");

	m_cbFaceCount.GridDimension = m_voxelTexDimensions;
	m_cbFaceCount.CurrentPhase = 0;
	m_cbFaceCount.VoxelCount = m_numberOfVoxels;

	compactBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	m_bufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.FlushResourceBarriers();

	VisibilityPass(context, DirectX::XMUINT3(ceil(m_numberOfVoxels / 256), 1, 1), compactBufferManager);
	context.Flush();

	m_cbFaceCount.CurrentPhase = 1;

	m_numberOfFaces = *m_bufferManager.ReadFromBuffer<UINT32*>(context, 1);
	m_bufferManager.ResizeBuffer(0, m_numberOfFaces);
	m_bufferManager.ZeroBuffer(context, 1);

	VisibilityPass(context, DirectX::XMUINT3(ceil(m_numberOfFaces / 256), 1, 1), compactBufferManager);

	PIXEndEvent(context.m_commandList->Get());

	context.Finish();
}

void CVGI::ClusterVisibility::VisibilityPass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize, BufferManager& compactBufferManager)
{
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[FaceCountPsoName].get());

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)FaceCountRootSignature::FaceCountCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbFaceCount).GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)FaceCountRootSignature::CompactSRVTable, compactBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)FaceCountRootSignature::FaceCountUAVTable, m_bufferManager.GetUAVHandle());

	context.Dispatch(groupSize.x, groupSize.y, groupSize.z);
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::ClusterVisibility::BuildFaceCountRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> clusterRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)FaceCountRootSignature::Count, 1);
	clusterRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*clusterRootSignature)[(UINT)FaceCountRootSignature::FaceCountCBV].InitAsConstantBuffer(0);
	(*clusterRootSignature)[(UINT)FaceCountRootSignature::CompactSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 0);
	(*clusterRootSignature)[(UINT)FaceCountRootSignature::FaceCountUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 0);

	clusterRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

	return clusterRootSignature;
}


std::shared_ptr<DX12Lib::ComputePipelineState> CVGI::ClusterVisibility::BuildFaceCountPipelineState(std::shared_ptr<DX12Lib::RootSignature> rootSig)
{
	std::wstring shaderPath = Utils::ToWstring(SOURCE_DIR) + L"\\Shaders";
	std::wstring computeShaderPath = shaderPath + L"\\FaceCount_CS.hlsl";

	std::shared_ptr<Shader> computeShader = std::make_shared<Shader>(computeShaderPath, "CS", "cs_5_1");
	computeShader->Compile();

	std::shared_ptr<ComputePipelineState> voxelClusterizeComputePso = std::make_shared<ComputePipelineState>();
	voxelClusterizeComputePso->SetRootSignature(rootSig);
	voxelClusterizeComputePso->SetComputeShader(computeShader);
	voxelClusterizeComputePso->Finalize();
	voxelClusterizeComputePso->Name = FaceCountPsoName;

	return voxelClusterizeComputePso;
}
