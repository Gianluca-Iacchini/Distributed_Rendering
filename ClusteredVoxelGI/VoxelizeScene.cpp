#include "VoxelizeScene.h"
#include "DX12Lib/pch.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"


#include "WinPixEventRuntime/pix3.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;


void CVGI::VoxelizeScene::InitializeBuffers()
{
	auto voxelGridSize = m_data->GetVoxelGridSize();

	UINT32 voxelSceneLinearSize = voxelGridSize.x * voxelGridSize.y * voxelGridSize.z;

	// Using bit representation of voxels to store whether a voxel is occupied or not.
	// Each bit represents wheter a specific voxel is occupied or not. We can store 32 voxels in a single UINT32,
	// So we divide the total number of voxels by 32 to get the size of the buffer.
	UINT32 voxelOccupiedSize = (voxelSceneLinearSize + 31) / 32;

	// Dummy buffers which will be resized once we know the number of fragments and voxels.
	// Voxel Occupied Buffer (u0)
	m_bufferManager->AddByteAddressBuffer(voxelOccupiedSize);
	// Fragment data Buffer (u1)
	m_bufferManager->AddStructuredBuffer(1, sizeof(FragmentData));
	// Next fragment index Buffer (u2)
	m_bufferManager->AddStructuredBuffer(1, sizeof(UINT32));
	// Voxel Index Buffer (u3)
	m_bufferManager->AddStructuredBuffer(voxelSceneLinearSize, sizeof(UINT32));
	// Fragment Counter Buffer (u4)
	m_bufferManager->AddByteAddressBuffer();
	// Voxel Counter Buffer (u5)
	m_bufferManager->AddByteAddressBuffer();
	// Voxel Hash Buffer (u6)
	m_bufferManager->AddStructuredBuffer(1, sizeof(UINT32));

	m_bufferManager->AllocateBuffers();
}

void CVGI::VoxelizeScene::UpdateBuffers(DX12Lib::CommandContext& context)
{
	auto voxelGridSize = m_data->GetVoxelGridSize();

	UINT32 fragmentCount = *m_bufferManager->ReadFromBuffer<UINT32*>(context, (UINT)VoxelBufferType::FragmentCounter);

	m_bufferManager->ResizeBuffer((UINT)VoxelBufferType::FragmentData, fragmentCount);
	m_bufferManager->ResizeBuffer((UINT)VoxelBufferType::NextIndex, fragmentCount);
	m_bufferManager->ResizeBuffer((UINT)VoxelBufferType::HashedBuffer, fragmentCount);
	m_data->FragmentCount = fragmentCount;


	UINT32 voxelLinearSize = voxelGridSize.x * voxelGridSize.y * voxelGridSize.z;

	UploadBuffer voxelIndexUploader;
	voxelIndexUploader.Create(voxelLinearSize * sizeof(UINT32));

	void* mappedData = voxelIndexUploader.Map();

	for (UINT32 i = 0; i < voxelLinearSize; i++)
	{
		((UINT32*)mappedData)[i] = UINT32_MAX;
	}

	DX12Lib::GPUBuffer& voxelIndexBuffer = m_bufferManager->GetBuffer((UINT)VoxelBufferType::VoxelIndex);

	context.TransitionResource(voxelIndexBuffer, D3D12_RESOURCE_STATE_COPY_DEST, true);
	context.m_commandList->Get()->CopyResource(voxelIndexBuffer.Get(), voxelIndexUploader.Get());

	m_bufferManager->ZeroBuffer(context, (UINT)VoxelBufferType::FragmentCounter);

	// We can unmap the buffer now because ZeroBuffer will flush the command queue
	voxelIndexUploader.Unmap();
}


void CVGI::VoxelizeScene::SetVoxelCamera(VoxelCamera* camera)
{
	assert(camera != nullptr);
	m_voxelCamera = camera;
}

void CVGI::VoxelizeScene::PerformTechnique(DX12Lib::GraphicsContext& context)
{
	assert(m_voxelCamera != nullptr && m_voxelCamera->IsEnabled);
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(0, 0, 128), Name.c_str());

	auto voxelSceneSize = m_data->GetVoxelGridSize();

	m_voxelScreenViewport = CD3DX12_VIEWPORT(0.0f, 0.0f, (float)voxelSceneSize.x, (float)voxelSceneSize.y, 0.0f, 1.0f);
	m_voxelScissorRect = CD3DX12_RECT(0, 0, (LONG)voxelSceneSize.x, (LONG)voxelSceneSize.y);

	TechniquePass(context);

	UpdateBuffers(context);

	m_currentPass = 1;
	TechniquePass(context);


	UINT32 voxelCount = *m_data->GetBufferManager(Name).ReadFromBuffer<UINT32*>(context, (UINT)VoxelBufferType::VoxelCounter);
	m_data->SetVoxelCount(voxelCount);

	PIXEndEvent(context.m_commandList->Get());
}

void CVGI::VoxelizeScene::TechniquePass(DX12Lib::GraphicsContext& context)
{
	auto& currentBackBuffer = Renderer::GetCurrentBackBuffer();

	context.TransitionResource(currentBackBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET);


	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);

	context.FlushResourceBarriers();

	context.ClearColor(currentBackBuffer, Color::LightSteelBlue().GetPtr(), nullptr);
	context.ClearDepthAndStencil(*Renderer::s_depthStencilBuffer);

	//context.SetRenderTargets(0, nullptr, Renderer::s_depthStencilBuffer->GetDSV());
	context.SetRenderTargets(1, &currentBackBuffer.GetRTV(), Renderer::s_depthStencilBuffer->GetDSV());

	context.SetViewportAndScissor(m_voxelScreenViewport, m_voxelScissorRect);

	context.SetPipelineState(Renderer::s_PSOs[Name].get());

	context.m_commandList->GetComPtr()->SetGraphicsRootConstantBufferView(
		(UINT)VoxelizeSceneRootParameterSlot::VoxelCommonCBV, m_data->GetVoxelCommonsResource().GpuAddress());

	assert(m_voxelCamera != nullptr && m_voxelCamera->IsEnabled);


	context.m_commandList->Get()->SetGraphicsRootConstantBufferView(
		(UINT)VoxelizeSceneRootParameterSlot::VoxelCameraCBV, m_voxelCamera->GetCameraBuffer().GpuAddress());
	
	context.m_commandList->Get()->SetGraphicsRoot32BitConstant((UINT)VoxelizeSceneRootParameterSlot::VoxelConstantCBV, m_currentPass, 0);

	context.m_commandList->Get()->SetGraphicsRootShaderResourceView(
		(UINT)VoxelizeSceneRootParameterSlot::MaterialSRV, Renderer::s_materialManager->GetMaterialStructuredBuffer().GpuAddress()
	);

	context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
		(UINT)VoxelizeSceneRootParameterSlot::VoxelDataUAV, m_bufferManager->GetUAVHandle());


	auto modelRenderers = Renderer::GetRenderers();

	for (ModelRenderer* mRenderer : modelRenderers)
	{
		auto batch = mRenderer->GetAll();

		mRenderer->Model->UseBuffers(context);

		for (auto& mesh : batch)
		{
			context.m_commandList->Get()->SetGraphicsRootDescriptorTable((UINT)VoxelizeSceneRootParameterSlot::MaterialTextureSRV, mesh->GetMaterialTextureSRV());
			context.m_commandList->Get()->SetGraphicsRootConstantBufferView((UINT)VoxelizeSceneRootParameterSlot::ObjectCBV, mesh->GetObjectCB().GpuAddress());
			mesh->DrawMesh(context);
		}
	}

}


void CVGI::VoxelizeScene::DeleteTemporaryBuffers()
{
	auto& bufferManager = m_data->GetBufferManager(Name);

	bufferManager.RemoveBuffer((UINT)VoxelBufferType::FragmentCounter);
	bufferManager.RemoveBuffer((UINT)VoxelBufferType::VoxelCounter);
	bufferManager.RemoveBuffer((UINT)VoxelBufferType::VoxelIndex);
	bufferManager.RemoveBuffer((UINT)VoxelBufferType::HashedBuffer);
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::VoxelizeScene::BuildRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> voxelizeSceneRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)VoxelizeSceneRootParameterSlot::Count, 1);
	voxelizeSceneRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*voxelizeSceneRootSignature)[(UINT)VoxelizeSceneRootParameterSlot::VoxelCommonCBV].InitAsConstantBuffer(0);
	(*voxelizeSceneRootSignature)[(UINT)VoxelizeSceneRootParameterSlot::VoxelCameraCBV].InitAsConstantBuffer(1);
	(*voxelizeSceneRootSignature)[(UINT)VoxelizeSceneRootParameterSlot::ObjectCBV].InitAsConstantBuffer(2);
	(*voxelizeSceneRootSignature)[(UINT)VoxelizeSceneRootParameterSlot::VoxelConstantCBV].InitAsConstants(3, 1);
	(*voxelizeSceneRootSignature)[(UINT)VoxelizeSceneRootParameterSlot::MaterialSRV].InitAsBufferSRV(0, D3D12_SHADER_VISIBILITY_PIXEL, 1);
	(*voxelizeSceneRootSignature)[(UINT)VoxelizeSceneRootParameterSlot::MaterialTextureSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, NUM_PBR_TEXTURES);
	(*voxelizeSceneRootSignature)[(UINT)VoxelizeSceneRootParameterSlot::VoxelDataUAV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 7);
	voxelizeSceneRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

	return voxelizeSceneRootSignature;
}

std::shared_ptr<DX12Lib::PipelineState> CVGI::VoxelizeScene::BuildPipelineState()
{
	std::shared_ptr<DX12Lib::RootSignature> voxelRootSig = BuildRootSignature();

	std::wstring shaderPath = Utils::ToWstring(SOURCE_DIR) + L"\\Shaders";
	std::wstring vertexShaderPath = shaderPath + L"\\Voxel_VS.hlsl";
	std::wstring geometryShaderPath = shaderPath + L"\\Voxel_GS.hlsl";
	std::wstring pixelShaderPath = shaderPath + L"\\Voxel_PS.hlsl";

	std::shared_ptr<Shader> vertexShader = std::make_shared<Shader>(vertexShaderPath, "VS", "vs_5_1");
	std::shared_ptr<Shader> geometryShader = std::make_shared<Shader>(geometryShaderPath, "GS", "gs_5_1");
	std::shared_ptr<Shader> pixelShader = std::make_shared<Shader>(pixelShaderPath, "PS", "ps_5_1");

	vertexShader->Compile();
	geometryShader->Compile();
	pixelShader->Compile();

	D3D12_RASTERIZER_DESC rasterizerDesc = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
	rasterizerDesc.CullMode = D3D12_CULL_MODE_NONE;

	D3D12_DEPTH_STENCIL_DESC depthStencilDesc = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
	depthStencilDesc.DepthEnable = FALSE;
	depthStencilDesc.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
	depthStencilDesc.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
	depthStencilDesc.StencilEnable = FALSE;

	std::shared_ptr<GraphicsPipelineState> voxelPSO = std::make_shared<GraphicsPipelineState>();
	voxelPSO->InitializeDefaultStates();
	voxelPSO->SetRasterizerState(rasterizerDesc);
	voxelPSO->SetDepthStencilState(depthStencilDesc);
	voxelPSO->SetInputLayout(DirectX::VertexPositionNormalTexture::InputLayout.pInputElementDescs, \
		DirectX::VertexPositionNormalTexture::InputLayout.NumElements);
	voxelPSO->SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
	voxelPSO->SetRenderTargetFormat(Graphics::m_backBufferFormat, Graphics::m_depthStencilFormat, 1, 0);
	//voxelPSO->SetRenderTargetFormats(0, nullptr, m_depthStencilFormat);
	voxelPSO->SetShader(vertexShader, ShaderType::Vertex);
	voxelPSO->SetShader(geometryShader, ShaderType::Geometry);
	voxelPSO->SetShader(pixelShader, ShaderType::Pixel);
	voxelPSO->SetRootSignature(voxelRootSig);
	voxelPSO->Name = Name;
	voxelPSO->Finalize();

	return voxelPSO;
}

const std::wstring CVGI::VoxelizeScene::Name = L"VoxelizeScene";


