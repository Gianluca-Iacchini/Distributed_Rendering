#include <DX12Lib/pch.h>
#include "ClusteredVoxelGI.h"
#include "VoxelScene.h"
#include "WinPixEventRuntime/pix3.h"
#include "VoxelCamera.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"



using namespace DirectX;
using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;

using namespace CVGI;



void ClusteredVoxelGIApp::Initialize(GraphicsContext& commandContext)
{

	auto voxelSceneRootSig = BuildVoxelizeSceneRootSignature();
	auto voxelScenePSO = BuildVoxelizeScenePso(voxelSceneRootSig);

	auto voxelDisplayRootSig = BuildVoxelDisplayRootSignature();
	auto voxelDisplayPSO = BuildVoxelDisplayPso(voxelDisplayRootSig);

	auto compactBufferRootSig = m_voxelBufferManager.BuildCompactBufferRootSignature();
	auto compactBufferPSO = m_voxelBufferManager.BuildCompactBufferPso(compactBufferRootSig);

	auto clusterizeRootSignature = m_voxelBufferManager.BuildClusterizeRootSignature();
	auto clusterizeVoxelPso = m_voxelBufferManager.BuildClulsterizePso(clusterizeRootSignature);

	Renderer::s_PSOs[voxelScenePSO->Name] = voxelScenePSO;
	Renderer::s_PSOs[voxelDisplayPSO->Name] = voxelDisplayPSO;
	Renderer::s_PSOs[compactBufferPSO->Name] = compactBufferPSO;
	Renderer::s_PSOs[clusterizeVoxelPso->Name] = clusterizeVoxelPso;

	m_voxelBufferManager.SetupFirstVoxelPassBuffers(VoxelTextureDimension);




	DirectX::XMFLOAT3 voxelTexDimensions = VoxelTextureDimension;
	DirectX::XMFLOAT3 voxelCellSize = DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f);

	VoxelScene* voxelScene = static_cast<VoxelScene*>(this->m_Scene.get());

	if (voxelScene != nullptr)
	{
		voxelScene->VoxelTextureDimensions = voxelTexDimensions;
		voxelScene->VoxelCellSize = voxelCellSize;
	}

	m_cbVoxelCommons.voxelTextureDimensions = DirectX::XMUINT3(voxelTexDimensions.x, voxelTexDimensions.y, voxelTexDimensions.z);
	m_cbVoxelCommons.voxelCellSize = voxelCellSize;
	m_cbVoxelCommons.invVoxelTextureDimensions = XMFLOAT3(1.0f / voxelTexDimensions.x, 1.0f / voxelTexDimensions.y, 1.0f / voxelTexDimensions.z);
	m_cbVoxelCommons.invVoxelCellSize = XMFLOAT3(1.0f / voxelCellSize.x, 1.0f / voxelCellSize.y, 1.0f / voxelCellSize.z);
	
	m_voxelScreenViewport.TopLeftX = 0;
	m_voxelScreenViewport.TopLeftY = 0;
	m_voxelScreenViewport.Width = voxelTexDimensions.x;
	m_voxelScreenViewport.Height = voxelTexDimensions.y;
	m_voxelScreenViewport.MinDepth = 0.0f;
	m_voxelScreenViewport.MaxDepth = 1.0f;

	m_voxelScissorRect = { 0, 0, (LONG)voxelTexDimensions.x, (LONG)voxelTexDimensions.y};


	D3DApp::Initialize(commandContext);


	Renderer::SetUpRenderFrame(commandContext);

	VoxelCamera* voxelCamera = voxelScene->GetVoxelCamera();
	voxelScene->Render(commandContext);
	if (voxelCamera != nullptr)
	{
		VoxelPass(commandContext, voxelCamera);
	}




	ReadBackBuffer voxelCounterReadBackBuffer;
	voxelCounterReadBackBuffer.Create(1, sizeof(UINT32));


	UINT32* fragmentCount = m_voxelBufferManager.ReadFromBuffer<UINT32*>(commandContext, BufferType::FragmentCounter);

	m_voxelBufferManager.SetupSecondVoxelPassBuffers(commandContext, *fragmentCount);

	m_cbVoxelCommons.StoreData = 1;

	if (voxelCamera != nullptr)
	{
		VoxelPass(commandContext, voxelCamera);
	}

	UINT32* voxelCount = m_voxelBufferManager.ReadFromBuffer<UINT32*>(commandContext, BufferType::VoxelCounter);

	DXLIB_CORE_INFO("Voxel Count: {0}", *voxelCount);

	m_voxelBufferManager.SetupCompactBuffers();

	m_voxelBufferManager.CompactBuffers();


	m_numberOfVoxels = m_voxelBufferManager.GetNumberOfVoxels();

	m_voxelBufferManager.InitializeClusters();
	m_voxelBufferManager.ClusterizeBuffers();

	UploadBuffer vertexUploadBuffer;
	vertexUploadBuffer.Create(m_numberOfVoxels * sizeof(UINT32));
	m_vertexBuffer.Create(m_numberOfVoxels, sizeof(UINT32));

	void* mappedData = vertexUploadBuffer.Map();

	for (UINT32 i = 0; i < m_numberOfVoxels; i++)
	{
		((UINT32*)mappedData)[i] = i;
	}



	// Not using CommandContext.CopyBuffer because upload buffer should not be transitioned from the GENERIC_READ state
	commandContext.TransitionResource(m_vertexBuffer, D3D12_RESOURCE_STATE_COPY_DEST, true);
	commandContext.m_commandList->Get()->CopyResource(m_vertexBuffer.Get(), vertexUploadBuffer.Get());

	commandContext.Flush(true);

	vertexUploadBuffer.Unmap();

	Renderer::PostDrawCleanup(commandContext);
}

void CVGI::ClusteredVoxelGIApp::Update(DX12Lib::GraphicsContext& commandContext)
{
	D3DApp::Update(commandContext);

	m_cbVoxelCommons.totalTime = GameTime::GetTotalTime();
	m_cbVoxelCommons.deltaTime = GameTime::GetDeltaTime();

	m_cbVoxelCommonsResource = Renderer::s_graphicsMemory->AllocateConstant(m_cbVoxelCommons);
}

void CVGI::ClusteredVoxelGIApp::Draw(DX12Lib::GraphicsContext& commandContext)
{
	Renderer::SetUpRenderFrame(commandContext);

	VoxelDisplayPass(commandContext);

	Renderer::PostDrawCleanup(commandContext);
}

void ClusteredVoxelGIApp::VoxelPass(DX12Lib::GraphicsContext& context, VoxelCamera* voxelCamera)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(0, 0, 128), L"VoxelPass");

	auto& currentBackBuffer = Renderer::GetCurrentBackBuffer();

	context.TransitionResource(currentBackBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET);
	
	context.TransitionResource(*m_voxelBufferManager.GetBuffer(BufferType::FragmentCounter), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.TransitionResource(*m_voxelBufferManager.GetBuffer(BufferType::VoxelCounter), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.TransitionResource(*m_voxelBufferManager.GetBuffer(BufferType::VoxelOccupied), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.TransitionResource(*m_voxelBufferManager.GetBuffer(BufferType::VoxelIndex), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	if (m_cbVoxelCommons.StoreData > 0)
	{
		context.TransitionResource(*m_voxelBufferManager.GetBuffer(BufferType::FragmentData), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		context.TransitionResource(*m_voxelBufferManager.GetBuffer(BufferType::NextIndex), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		context.TransitionResource(*m_voxelBufferManager.GetBuffer(BufferType::HashedBuffer), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	}

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
		(UINT)VoxelizeSceneRootParameterSlot::VoxelDataUAV, m_voxelBufferManager.GetVoxelizeTableUAV());


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




void CVGI::ClusteredVoxelGIApp::VoxelDisplayPass(DX12Lib::GraphicsContext& context)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 128), L"Voxel Display Pass");

	auto& currentBackBuffer = Renderer::GetCurrentBackBuffer();

	for (auto* buffer : m_uavBuffers)
	{
		context.TransitionResource(*buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	}

	context.TransitionResource(m_vertexBuffer, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
	context.TransitionResource(currentBackBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, true);
	context.TransitionResource(*m_voxelBufferManager.GetBuffer(BufferType::ClusterData), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

	context.ClearColor(currentBackBuffer, Color::LightSteelBlue().GetPtr(), nullptr);
	context.ClearDepthAndStencil(*Renderer::s_depthStencilBuffer);

	context.SetRenderTargets(1, &currentBackBuffer.GetRTV(), Renderer::s_depthStencilBuffer->GetDSV());

	context.SetViewportAndScissor(Renderer::s_screenViewport, Renderer::s_scissorRect);

	context.SetPipelineState(Renderer::s_PSOs[L"PSO_DISPLAY_VOXEL"].get());

	context.m_commandList->Get()->SetGraphicsRootConstantBufferView((UINT)DisplayVoxelRootParameterSlot::VoxelCommonCBV,
				m_cbVoxelCommonsResource.GpuAddress());

	context.m_commandList->Get()->SetGraphicsRootConstantBufferView((UINT)DisplayVoxelRootParameterSlot::CameraCBV,
		m_Scene->GetMainCamera()->GetCameraBuffer().GpuAddress());


	context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
		(UINT)DisplayVoxelRootParameterSlot::VoxelSRVBufferTable, m_voxelBufferManager.GetVoxelizeTableSRV());
	
	context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
		(UINT)DisplayVoxelRootParameterSlot::CompactSRVBufferTable, m_voxelBufferManager.GetCompactionTableSRV());

	context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
		(UINT)DisplayVoxelRootParameterSlot::ClusterSRVBufferTable, m_voxelBufferManager.GetClusterizeTableSRV());

	context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
		(UINT)DisplayVoxelRootParameterSlot::ClusterUAVBufferTable, m_voxelBufferManager.GetClusterizeTableUAV());

	context.m_commandList->Get()->IASetVertexBuffers(0, 1, &m_vertexBuffer.VertexBufferView());
	context.m_commandList->Get()->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);

	context.m_commandList->Get()->DrawInstanced(m_numberOfVoxels, 1, 0, 0);
	
	PIXEndEvent(context.m_commandList->Get());
}


std::shared_ptr<DX12Lib::RootSignature> ClusteredVoxelGIApp::BuildVoxelizeSceneRootSignature()
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
	//(*voxelizeSceneRootSignature)[(UINT)VoxelizeSceneRootParameterSlot::VoxelDataUAV].InitAsBufferUAV(0, D3D12_SHADER_VISIBILITY_ALL, 1);
	//(*voxelizeSceneRootSignature)[(UINT)VoxelizeSceneRootParameterSlot::VoxelDataCounterUAV].InitAsBufferUAV(1, D3D12_SHADER_VISIBILITY_ALL, 1);
	voxelizeSceneRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

	return voxelizeSceneRootSignature;
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::ClusteredVoxelGIApp::BuildVoxelDisplayRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> displayVoxelRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)DisplayVoxelRootParameterSlot::Count, 1);
	displayVoxelRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::VoxelCommonCBV].InitAsConstantBuffer(0);
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::CameraCBV].InitAsConstantBuffer(1);
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::VoxelSRVBufferTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 2);
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::CompactSRVBufferTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 1);
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::ClusterSRVBufferTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 3, D3D12_SHADER_VISIBILITY_ALL, 2);
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::ClusterUAVBufferTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 8, D3D12_SHADER_VISIBILITY_ALL, 0);
	displayVoxelRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

	return displayVoxelRootSignature;
}


std::shared_ptr<DX12Lib::GraphicsPipelineState> ClusteredVoxelGIApp::BuildVoxelizeScenePso(std::shared_ptr<DX12Lib::RootSignature> voxelRootSig)
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
	voxelPSO->SetRenderTargetFormat(m_backBufferFormat, m_depthStencilFormat, 1, 0);
	//voxelPSO->SetRenderTargetFormats(0, nullptr, m_depthStencilFormat);
	voxelPSO->SetShader(vertexShader, ShaderType::Vertex);
	voxelPSO->SetShader(geometryShader, ShaderType::Geometry);
	voxelPSO->SetShader(pixelShader, ShaderType::Pixel);
	voxelPSO->SetRootSignature(voxelRootSig);
	voxelPSO->Name = L"PSO_VOXELIZE_SCENE";
	voxelPSO->Finalize();

	return voxelPSO;
}

std::shared_ptr<DX12Lib::GraphicsPipelineState> CVGI::ClusteredVoxelGIApp::BuildVoxelDisplayPso(std::shared_ptr<DX12Lib::RootSignature> voxelRootSig)
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
	displayVoxelPSO->SetInputLayout(CVGI::VertexSingleUINT::InputLayout.pInputElementDescs, \
		CVGI::VertexSingleUINT::InputLayout.NumElements);
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


	




int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance, PSTR cmdLine, int showCmd)
{
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
	try
	{
		CVGI::ClusteredVoxelGIApp app(hInstance, new CVGI::VoxelScene());

		if (!app.InitializeApp())
			return 0;

		return app.Run();
	}
	catch (DxException& e)
	{
		MessageBox(nullptr, e.ToString().c_str(), L"HR Failed", MB_OK);
		return 0;
	}
}