#include "VoxelizeScene.h"
#include "DX12Lib/pch.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"


#include "WinPixEventRuntime/pix3.h"

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;

CVGI::VoxelizeScene::VoxelizeScene(DirectX::XMUINT3 VoxelSceneSize, DirectX::XMFLOAT3 VoxelSize) 
	: m_voxelSceneDimensions(VoxelSceneSize), m_vertexBuffer(DXGI_FORMAT_R32_UINT)
{
	m_voxelScreenViewport = CD3DX12_VIEWPORT(0.0f, 0.0f, (float)VoxelSceneSize.x, (float)VoxelSceneSize.y, 0.0f, 1.0f);
	m_voxelScissorRect = CD3DX12_RECT(0, 0, (LONG)VoxelSceneSize.x, (LONG)VoxelSceneSize.y);

	m_cbVoxelCommons.voxelTextureDimensions = DirectX::XMUINT3(VoxelSceneSize.x, VoxelSceneSize.y, VoxelSceneSize.z);
	m_cbVoxelCommons.invVoxelTextureDimensions = DirectX::XMFLOAT3(1.0f / VoxelSceneSize.x, 1.0f / VoxelSceneSize.y, 1.0f / VoxelSceneSize.z);
	m_cbVoxelCommons.voxelCellSize = VoxelSize;
	m_cbVoxelCommons.invVoxelCellSize = DirectX::XMFLOAT3(1.0f / VoxelSize.x, 1.0f / VoxelSize.y, 1.0f / VoxelSize.z);
}

void VoxelizeScene::InitializeBuffers()
{
	UINT32 voxelSceneLinearSize = m_voxelSceneDimensions.x * m_voxelSceneDimensions.y * m_voxelSceneDimensions.z;

	// Using bit representation of voxels to store whether a voxel is occupied or not.
	// Each bit represents wheter a specific voxel is occupied or not. We can store 32 voxels in a single UINT32,
	// So we divide the total number of voxels by 32 to get the size of the buffer.
	UINT32 voxelOccupiedSize = (voxelSceneLinearSize + 31) / 32;

	// Dummy buffers which will be resized once we know the number of fragments and voxels.
	// Fragment data Buffer (u0)
	m_bufferManager.AddStructuredBuffer(1, sizeof(FragmentData));
	// Next fragment index Buffer (u1)
	m_bufferManager.AddStructuredBuffer(1, sizeof(UINT32));
	// Voxel Index Buffer (u2)
	m_bufferManager.AddStructuredBuffer(voxelSceneLinearSize, sizeof(UINT32));
	// Fragment Counter Buffer (u3)
	m_bufferManager.AddByteAddressBuffer();
	// Voxel Counter Buffer (u4)
	m_bufferManager.AddByteAddressBuffer();
	// Voxel Occupied Buffer (u5)
	m_bufferManager.AddStructuredBuffer(voxelOccupiedSize, sizeof(UINT32));
	// Voxel Hash Buffer (u6)
	m_bufferManager.AddStructuredBuffer(1, sizeof(UINT32));

	m_bufferManager.AllocateBuffers();
}

void CVGI::VoxelizeScene::UpdateBuffers(DX12Lib::CommandContext& context)
{
	UINT32* fragmentCount = m_bufferManager.ReadFromBuffer<UINT32*>(context, (UINT)VoxelBufferType::FragmentCounter);

	m_fragmentCount = *fragmentCount;

	m_bufferManager.ResizeBuffer((UINT)VoxelBufferType::FragmentData, m_fragmentCount);
	m_bufferManager.ResizeBuffer((UINT)VoxelBufferType::NextIndex, m_fragmentCount);
	m_bufferManager.ResizeBuffer((UINT)VoxelBufferType::HashedBuffer, m_fragmentCount);



	UINT32 voxelLinearSize = m_voxelSceneDimensions.x * m_voxelSceneDimensions.y * m_voxelSceneDimensions.z;

	UploadBuffer voxelIndexUploader;
	voxelIndexUploader.Create(voxelLinearSize * sizeof(UINT32));

	void* mappedData = voxelIndexUploader.Map();

	for (UINT32 i = 0; i < voxelLinearSize; i++)
	{
		((UINT32*)mappedData)[i] = UINT32_MAX;
	}

	DX12Lib::GPUBuffer& voxelIndexBuffer = m_bufferManager.GetBuffer((UINT)VoxelBufferType::VoxelIndex);

	context.TransitionResource(voxelIndexBuffer, D3D12_RESOURCE_STATE_COPY_DEST, true);
	context.m_commandList->Get()->CopyResource(voxelIndexBuffer.Get(), voxelIndexUploader.Get());

	m_bufferManager.ZeroBuffer(context, (UINT)VoxelBufferType::FragmentCounter);

	// We can unmap the buffer now because ZeroBuffer will flush the command queue
	voxelIndexUploader.Unmap();

	m_cbVoxelCommons.StoreData = 1;
}

void CVGI::VoxelizeScene::SetVertexData(DX12Lib::CommandContext& context, UINT32 vertexCount)
{

	UploadBuffer vertexUploadBuffer;
	vertexUploadBuffer.Create(vertexCount * sizeof(UINT32));
	m_vertexBuffer.Create(vertexCount, sizeof(UINT32));

	void* mappedData = vertexUploadBuffer.Map();

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

void VoxelizeScene::VoxelizePass(DX12Lib::GraphicsContext& context, VoxelCamera* voxelCamera)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(0, 0, 128), L"VoxelPass");

	auto& currentBackBuffer = Renderer::GetCurrentBackBuffer();

	context.TransitionResource(currentBackBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET);

	m_bufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);

	context.FlushResourceBarriers();

	context.ClearColor(currentBackBuffer, Color::LightSteelBlue().GetPtr(), nullptr);
	context.ClearDepthAndStencil(*Renderer::s_depthStencilBuffer);

	//context.SetRenderTargets(0, nullptr, Renderer::s_depthStencilBuffer->GetDSV());
	context.SetRenderTargets(1, &currentBackBuffer.GetRTV(), Renderer::s_depthStencilBuffer->GetDSV());

	context.SetViewportAndScissor(m_voxelScreenViewport, m_voxelScissorRect);

	context.SetPipelineState(Renderer::s_PSOs[L"PSO_VOXELIZE_SCENE"].get());

	m_cbVoxelCommons.totalTime = GameTime::GetTotalTime();
	m_cbVoxelCommons.deltaTime = GameTime::GetDeltaTime();

	context.m_commandList->GetComPtr()->SetGraphicsRootConstantBufferView(
		(UINT)VoxelizeSceneRootParameterSlot::VoxelCommonCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbVoxelCommons).GpuAddress());

	if (voxelCamera->IsEnabled)
	{
		context.m_commandList->Get()->SetGraphicsRootConstantBufferView(
			(UINT)VoxelizeSceneRootParameterSlot::VoxelCameraCBV, voxelCamera->GetCameraBuffer().GpuAddress());
	}


	context.m_commandList->Get()->SetGraphicsRootShaderResourceView(
		(UINT)VoxelizeSceneRootParameterSlot::MaterialSRV, Renderer::s_materialManager->GetMaterialStructuredBuffer().GpuAddress()
	);

	context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
		(UINT)VoxelizeSceneRootParameterSlot::VoxelDataUAV, m_bufferManager.GetUAVHandle());


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

	PIXEndEvent(context.m_commandList->Get());
}

void CVGI::VoxelizeScene::DisplayVoxelPass(DX12Lib::GraphicsContext& context, DX12Lib::SceneCamera* camera, BufferManager* compactBufferManager, BufferManager* clusterBufferManager)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 128), L"Voxel Display Pass");

	m_cbVoxelCommons.deltaTime = GameTime::GetDeltaTime();
	m_cbVoxelCommons.totalTime = GameTime::GetTotalTime();

	auto& currentBackBuffer = Renderer::GetCurrentBackBuffer();

	if (compactBufferManager != nullptr)
	{
		compactBufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	}
	if (clusterBufferManager != nullptr)
	{
		clusterBufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	}

	context.TransitionResource(m_vertexBuffer, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
	context.TransitionResource(currentBackBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, true);

	context.ClearColor(currentBackBuffer, Color::LightSteelBlue().GetPtr(), nullptr);
	context.ClearDepthAndStencil(*Renderer::s_depthStencilBuffer);

	context.SetRenderTargets(1, &currentBackBuffer.GetRTV(), Renderer::s_depthStencilBuffer->GetDSV());

	context.SetViewportAndScissor(Renderer::s_screenViewport, Renderer::s_scissorRect);

	context.SetPipelineState(Renderer::s_PSOs[L"PSO_DISPLAY_VOXEL"].get());

	context.m_commandList->Get()->SetGraphicsRootConstantBufferView((UINT)DisplayVoxelRootParameterSlot::VoxelCommonCBV,
		Renderer::s_graphicsMemory->AllocateConstant(m_cbVoxelCommons).GpuAddress());

	context.m_commandList->Get()->SetGraphicsRootConstantBufferView((UINT)DisplayVoxelRootParameterSlot::CameraCBV,
		camera->GetCameraBuffer().GpuAddress());


	context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
		(UINT)DisplayVoxelRootParameterSlot::VoxelSRVBufferTable, m_bufferManager.GetSRVHandle());

	if (compactBufferManager != nullptr)
	{
		context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
			(UINT)DisplayVoxelRootParameterSlot::CompactSRVBufferTable, compactBufferManager->GetSRVHandle());
	}

	if (clusterBufferManager != nullptr)
	{
		context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
			(UINT)DisplayVoxelRootParameterSlot::ClusterSRVBufferTable, clusterBufferManager->GetSRVHandle());
	}


	//context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
	//	(UINT)DisplayVoxelRootParameterSlot::ClusterUAVBufferTable, m_voxelBufferManager.GetClusterizeTableUAV());

	context.m_commandList->Get()->IASetVertexBuffers(0, 1, &m_vertexBuffer.VertexBufferView());
	context.m_commandList->Get()->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);

	context.m_commandList->Get()->DrawInstanced(m_vertexCount, 1, 0, 0);

	PIXEndEvent(context.m_commandList->Get());
}

void CVGI::VoxelizeScene::DeleteTemporaryBuffers()
{
	m_bufferManager.RemoveBuffer((UINT)VoxelBufferType::FragmentCounter);
	m_bufferManager.RemoveBuffer((UINT)VoxelBufferType::VoxelCounter);
	m_bufferManager.RemoveBuffer((UINT)VoxelBufferType::VoxelOccupied);
	m_bufferManager.RemoveBuffer((UINT)VoxelBufferType::VoxelIndex);
	m_bufferManager.RemoveBuffer((UINT)VoxelBufferType::HashedBuffer);
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::VoxelizeScene::BuildVoxelizeSceneRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> voxelizeSceneRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)VoxelizeSceneRootParameterSlot::Count, 1);
	voxelizeSceneRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*voxelizeSceneRootSignature)[(UINT)VoxelizeSceneRootParameterSlot::VoxelCommonCBV].InitAsConstantBuffer(0);
	(*voxelizeSceneRootSignature)[(UINT)VoxelizeSceneRootParameterSlot::VoxelCameraCBV].InitAsConstantBuffer(1);
	(*voxelizeSceneRootSignature)[(UINT)VoxelizeSceneRootParameterSlot::ObjectCBV].InitAsConstantBuffer(2);
	(*voxelizeSceneRootSignature)[(UINT)VoxelizeSceneRootParameterSlot::MaterialSRV].InitAsBufferSRV(0, D3D12_SHADER_VISIBILITY_PIXEL, 1);
	(*voxelizeSceneRootSignature)[(UINT)VoxelizeSceneRootParameterSlot::MaterialTextureSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, NUM_PBR_TEXTURES);
	(*voxelizeSceneRootSignature)[(UINT)VoxelizeSceneRootParameterSlot::VoxelDataUAV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 7);
	voxelizeSceneRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

	return voxelizeSceneRootSignature;
}

std::shared_ptr<DX12Lib::GraphicsPipelineState> CVGI::VoxelizeScene::BuildVoxelizeScenePso(std::shared_ptr<RootSignature> voxelRootSig)
{
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
	voxelPSO->Name = L"PSO_VOXELIZE_SCENE";
	voxelPSO->Finalize();

	return voxelPSO;
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::VoxelizeScene::BuildDisplayVoxelRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> displayVoxelRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)DisplayVoxelRootParameterSlot::Count - 1, 1);
	displayVoxelRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::VoxelCommonCBV].InitAsConstantBuffer(0);
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::CameraCBV].InitAsConstantBuffer(1);
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::VoxelSRVBufferTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 2);
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::CompactSRVBufferTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 1);
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::ClusterSRVBufferTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 3, D3D12_SHADER_VISIBILITY_ALL, 2);
	//(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::ClusterUAVBufferTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 8, D3D12_SHADER_VISIBILITY_ALL, 0);
	displayVoxelRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

	return displayVoxelRootSignature;
}

std::shared_ptr<DX12Lib::GraphicsPipelineState> CVGI::VoxelizeScene::BuildDisplayVoxelPso(std::shared_ptr<DX12Lib::RootSignature> voxelRootSig)
{
	std::wstring shaderPath = Utils::ToWstring(SOURCE_DIR) + L"\\Shaders";
	std::wstring vertexShaderPath = shaderPath + L"\\VoxelDisplay_VS.hlsl";
	std::wstring geometryShaderPath = shaderPath + L"\\VoxelDisplay_GS.hlsl";
	std::wstring pixelShaderPath = shaderPath + L"\\VoxelDisplay_PS.hlsl";

	std::shared_ptr<Shader> vertexShader = std::make_shared<Shader>(vertexShaderPath, "VS", "vs_5_1");
	std::shared_ptr<Shader> geometryShader = std::make_shared<Shader>(geometryShaderPath, "GS", "gs_5_1");
	std::shared_ptr<Shader> pixelShader = std::make_shared<Shader>(pixelShaderPath, "PS", "ps_5_1");

	vertexShader->Compile();
	geometryShader->Compile();
	pixelShader->Compile();

	std::shared_ptr<GraphicsPipelineState> displayVoxelPSO = std::make_shared<GraphicsPipelineState>();
	displayVoxelPSO->InitializeDefaultStates();
	displayVoxelPSO->SetInputLayout(CVGI::VoxelizeScene::VertexSingleUINT::InputLayout.pInputElementDescs, \
		CVGI::VoxelizeScene::VertexSingleUINT::InputLayout.NumElements);
	displayVoxelPSO->SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT);
	displayVoxelPSO->SetRenderTargetFormat(m_backBufferFormat, m_depthStencilFormat, 1, 0);
	displayVoxelPSO->SetShader(vertexShader, ShaderType::Vertex);
	displayVoxelPSO->SetShader(geometryShader, ShaderType::Geometry);
	displayVoxelPSO->SetShader(pixelShader, ShaderType::Pixel);
	displayVoxelPSO->SetRootSignature(voxelRootSig);
	displayVoxelPSO->Name = L"PSO_DISPLAY_VOXEL";
	displayVoxelPSO->Finalize();

	return displayVoxelPSO;
}


const D3D12_INPUT_ELEMENT_DESC VoxelizeScene::VertexSingleUINT::InputElements[] =
{
	{ "SV_Position", 0, DXGI_FORMAT_R32_UINT,    0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
};

static_assert(sizeof(VoxelizeScene::VertexSingleUINT) == 4, "Vertex struct/layout mismatch");

const D3D12_INPUT_LAYOUT_DESC VoxelizeScene::VertexSingleUINT::InputLayout =
{
	VertexSingleUINT::InputElements,
	VertexSingleUINT::InputElementCount
};