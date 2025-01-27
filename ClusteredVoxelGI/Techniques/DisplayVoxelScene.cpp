#include "DisplayVoxelScene.h"
#include "DX12Lib/pch.h"
#include "DX12Lib/Scene/SceneCamera.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"
#include "VoxelizeScene.h"
#include "PrefixSumVoxels.h"
#include "ClusterVoxels.h"
#include "LightVoxel.h"
#include "LightTransportTechnique.h"
#include "GaussianFilterTechnique.h"
#include "../Data/Shaders/Include/VoxelDisplay_VS.h"
#include "../Data/Shaders/Include/VoxelDisplay_GS.h"
#include "../Data/Shaders/Include/VoxelDisplay_PS.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;
using namespace VOX;

CVGI::DisplayVoxelScene::DisplayVoxelScene(std::shared_ptr<VOX::TechniqueData> data) : m_vertexBuffer(DXGI_FORMAT_R32_UINT)
{
	m_bufferManager = std::make_shared<BufferManager>();
	data->SetBufferManager(Name, m_bufferManager);
	m_data = data;
}

void CVGI::DisplayVoxelScene::InitializeBuffers()
{

}

void CVGI::DisplayVoxelScene::PerformTechnique(DX12Lib::GraphicsContext& context)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 128), L"Voxel Display Pass");

	TechniquePass(context);

	PIXEndEvent(context.m_commandList->Get());
}

void CVGI::DisplayVoxelScene::SetVertexData(DX12Lib::GraphicsContext& context)
{
	UINT32 vertexCount = m_data->FaceCount;

	UploadBuffer vertexUploadBuffer;
	vertexUploadBuffer.Create(vertexCount * sizeof(UINT32));
	m_vertexBuffer.Create(vertexCount, sizeof(UINT32));

	void* mappedData = vertexUploadBuffer.Map();

	// The vertex buffer is just an array of indices frin 0 to faceCount-1.
	// Not great but fine for a debug display.
	for (UINT32 i = 0; i < vertexCount; i++)
	{
		((UINT32*)mappedData)[i] = i;
	}

	// Not using CommandContext.CopyBuffer because upload buffer should not be transitioned from the GENERIC_READ state
	context.TransitionResource(m_vertexBuffer, D3D12_RESOURCE_STATE_COPY_DEST, true);
	context.m_commandList->Get()->CopyResource(m_vertexBuffer.Get(), vertexUploadBuffer.Get());

	context.Flush(true);

	m_vertexCount = vertexCount;
	vertexUploadBuffer.Unmap();
}

void CVGI::DisplayVoxelScene::SetCamera(DX12Lib::SceneCamera* camera)
{
	m_camera = camera;
}

void CVGI::DisplayVoxelScene::TechniquePass(DX12Lib::GraphicsContext& context)
{
	auto& currentBackBuffer = Graphics::Renderer::GetCurrentBackBuffer();

	auto& voxelBufferManager = m_data->GetBufferManager(VoxelizeScene::Name);
	auto& compactBufferManager = m_data->GetBufferManager(PrefixSumVoxels::Name);
	auto& clusterBufferManager = m_data->GetBufferManager(ClusterVoxels::Name);
	auto& shadowBufferManager = m_data->GetBufferManager(LightVoxel::Name);
	auto& lightTransportBufferManager = m_data->GetBufferManager(LightTransportTechnique::IndirectName);
	auto& gaussianBufferManager = m_data->GetBufferManager(GaussianFilterTechnique::ReadName);

	compactBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	clusterBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	shadowBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	lightTransportBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	gaussianBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	

	context.TransitionResource(m_vertexBuffer, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
	context.TransitionResource(currentBackBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, true);

	context.ClearColor(currentBackBuffer, Color::LightSteelBlue().GetPtr(), nullptr);
	context.ClearDepthAndStencil(*Renderer::s_depthStencilBuffer);

	context.SetRenderTargets(1, &currentBackBuffer.GetRTV(), Renderer::s_depthStencilBuffer->GetDSV());

	context.SetViewportAndScissor(Renderer::s_screenViewport, Renderer::s_scissorRect);

	context.SetPipelineState(m_techniquePSO.get());

	context.m_commandList->Get()->SetGraphicsRootConstantBufferView((UINT)DisplayVoxelRootParameterSlot::VoxelCommonCBV,
		m_data->GetVoxelCommonsResource().GpuAddress());

	context.m_commandList->Get()->SetGraphicsRootConstantBufferView((UINT)DisplayVoxelRootParameterSlot::CameraCBV,
		m_camera->GetCameraBuffer().GpuAddress());

	context.m_commandList->Get()->SetGraphicsRoot32BitConstants((UINT)DisplayVoxelRootParameterSlot::DisplayModeCBV, 1, &m_displayMode, 0);

	context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
		(UINT)DisplayVoxelRootParameterSlot::VoxelSRVBufferTable, voxelBufferManager.GetSRVHandle());

	context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
		(UINT)DisplayVoxelRootParameterSlot::CompactSRVBufferTable, compactBufferManager.GetSRVHandle());

	context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
		(UINT)DisplayVoxelRootParameterSlot::ClusterSRVBufferTable, clusterBufferManager.GetSRVHandle());

	context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
		(UINT)DisplayVoxelRootParameterSlot::ShadowSRVBufferTable, shadowBufferManager.GetSRVHandle());	
	
	context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
		(UINT)DisplayVoxelRootParameterSlot::LightTransportSRVBufferTable, lightTransportBufferManager.GetSRVHandle());
	
	context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
		(UINT)DisplayVoxelRootParameterSlot::GaussianSRVBufferTable, gaussianBufferManager.GetSRVHandle());


	D3D12_VERTEX_BUFFER_VIEW vertexBufferView = m_vertexBuffer.VertexBufferView();

	context.m_commandList->Get()->IASetVertexBuffers(0, 1, &vertexBufferView);
	context.m_commandList->Get()->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
	context.m_commandList->Get()->DrawInstanced(m_vertexCount, 1, 0, 0);


}

std::shared_ptr<DX12Lib::RootSignature> CVGI::DisplayVoxelScene::BuildRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> displayVoxelRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)DisplayVoxelRootParameterSlot::Count, 1);
	displayVoxelRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::VoxelCommonCBV].InitAsConstantBuffer(0);
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::CameraCBV].InitAsConstantBuffer(1);
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::DisplayModeCBV].InitAsConstants(2, 1);
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::VoxelSRVBufferTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 3);
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::CompactSRVBufferTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 1);
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::ClusterSRVBufferTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 3, D3D12_SHADER_VISIBILITY_ALL, 2);
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::ShadowSRVBufferTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 3);
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::LightTransportSRVBufferTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1, D3D12_SHADER_VISIBILITY_ALL, 4);
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::GaussianSRVBufferTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1, D3D12_SHADER_VISIBILITY_ALL, 5);
	displayVoxelRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

	return displayVoxelRootSignature;
}

void CVGI::DisplayVoxelScene::BuildPipelineState()
{
	std::shared_ptr<DX12Lib::RootSignature> voxelRootSig = BuildRootSignature();

	auto vertexShaderByteCode = CD3DX12_SHADER_BYTECODE((void*)g_pVoxelDisplay_VS, ARRAYSIZE(g_pVoxelDisplay_VS));
	auto geometryShaderByteCode = CD3DX12_SHADER_BYTECODE((void*)g_pVoxelDisplay_GS, ARRAYSIZE(g_pVoxelDisplay_GS));
	auto pixelShaderByteCode = CD3DX12_SHADER_BYTECODE((void*)g_pVoxelDisplay_PS, ARRAYSIZE(g_pVoxelDisplay_PS));

	std::unique_ptr<GraphicsPipelineState> displayVoxelPSO = std::make_unique<GraphicsPipelineState>();
	displayVoxelPSO->InitializeDefaultStates();
	displayVoxelPSO->SetInputLayout(VertexSingleUINT::InputLayout.pInputElementDescs, \
		VertexSingleUINT::InputLayout.NumElements);
	displayVoxelPSO->SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT);
	displayVoxelPSO->SetRenderTargetFormat(Graphics::m_backBufferFormat, Graphics::m_depthStencilFormat, 1, 0);
	displayVoxelPSO->SetShader(vertexShaderByteCode, ShaderType::Vertex);
	displayVoxelPSO->SetShader(geometryShaderByteCode, ShaderType::Geometry);
	displayVoxelPSO->SetShader(pixelShaderByteCode, ShaderType::Pixel);
	displayVoxelPSO->SetRootSignature(voxelRootSig);
	displayVoxelPSO->Name = Name;
	displayVoxelPSO->Finalize();

	m_techniquePSO = std::move(displayVoxelPSO);
}

const std::wstring CVGI::DisplayVoxelScene::Name = L"DisplayVoxelScene";

const D3D12_INPUT_ELEMENT_DESC DisplayVoxelScene::VertexSingleUINT::InputElements[] =
{
	{ "SV_Position", 0, DXGI_FORMAT_R32_UINT,    0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
};

static_assert(sizeof(DisplayVoxelScene::VertexSingleUINT) == 4, "Vertex struct/layout mismatch");

const D3D12_INPUT_LAYOUT_DESC DisplayVoxelScene::VertexSingleUINT::InputLayout =
{
	VertexSingleUINT::InputElements,
	VertexSingleUINT::InputElementCount
};