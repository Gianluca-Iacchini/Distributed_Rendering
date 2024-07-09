#include <DX12Lib/pch.h>
#include "ClusteredVoxelGI.h"
#include "VoxelScene.h"
#include "WinPixEventRuntime/pix3.h"
#include "VoxelCamera.h"



using namespace DirectX;
using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;

using namespace CVGI;



void ClusteredVoxelGIApp::Initialize(GraphicsContext& commandContext)
{
	//auto voxelComputeRootSig = BuildVoxelComputeRootSignature();
	//auto voxelComputePSO = BuildVoxelComputePso(voxelComputeRootSig);

	m_voxelTexture.Create3D(VoxelTextureDimension.x, VoxelTextureDimension.y, VoxelTextureDimension.z , 1, DXGI_FORMAT_R8G8B8A8_UINT);
	Renderer::SetVoxelTexture(m_voxelTexture);

	auto voxelSceneRootSig = BuildVoxelizeSceneRootSignature();
	auto voxelScenePSO = BuildVoxelizeScenePso(voxelSceneRootSig);

	auto voxelDisplayRootSig = BuildVoxelDisplayRootSignature();
	auto voxelDisplayPSO = BuildVoxelDisplayPso(voxelDisplayRootSig);

	//Renderer::s_PSOs[voxelComputePSO->Name] = voxelComputePSO;
	Renderer::s_PSOs[voxelScenePSO->Name] = voxelScenePSO;
	Renderer::s_PSOs[voxelDisplayPSO->Name] = voxelDisplayPSO;

	m_voxelDataBuffer.Create(VoxelTextureDimension.x * VoxelTextureDimension.y * VoxelTextureDimension.z, sizeof(VoxelUAVData));

	m_voxelDataUAVStart = Renderer::s_textureHeap->Alloc(1);

	Graphics::s_device->Get()->CopyDescriptorsSimple(1, m_voxelDataUAVStart, m_voxelDataBuffer.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, m_voxelDataUAVStart + Renderer::s_textureHeap->GetDescriptorSize(), m_voxelDataBuffer.GetCounterBuffer().GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	DirectX::XMFLOAT3 voxelTexDimensions = VoxelTextureDimension;
	DirectX::XMFLOAT3 voxelCellSize = DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f);

	VoxelScene* voxelScene = static_cast<VoxelScene*>(this->m_Scene.get());

	if (voxelScene != nullptr)
	{
		voxelScene->VoxelTextureDimensions = voxelTexDimensions;
		voxelScene->VoxelCellSize = voxelCellSize;
	}

	m_cbVoxelCommons.voxelTextureDimensions = voxelTexDimensions;
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

	InitializeVoxelDisplayBuffers();



	D3DApp::Initialize(commandContext);


	Renderer::SetUpRenderFrame(commandContext);

	VoxelCamera* voxelCamera = voxelScene->GetVoxelCamera();
	voxelScene->Render(commandContext);
	if (voxelCamera != nullptr)
	{
		VoxelPass(commandContext, voxelCamera);
	}

	Renderer::PostDrawCleanup(commandContext);
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

	context.TransitionResource(currentBackBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, true);

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
		(UINT)VoxelizeSceneRootParameterSlot::VoxelTextureUAV, Renderer::GetVoxelTextureUAV(context));

	context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
		(UINT)VoxelizeSceneRootParameterSlot::VoxelDataUAV, m_voxelDataUAVStart);


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

void CVGI::ClusteredVoxelGIApp::VoxelComputePass()
{
	ComputeContext& context = ComputeContext::Begin();

	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 0), L"VoxelComputePass");

	context.SetPipelineState(Renderer::s_PSOs[L"PSO_VOXELIZE_SCENE_COMPUTE"].get());
	
	context.Dispatch3D(VoxelTextureDimension.x, VoxelTextureDimension.y, VoxelTextureDimension.z, 8, 8, 8);

	PIXEndEvent(context.m_commandList->Get());

	context.Finish();
}

void CVGI::ClusteredVoxelGIApp::VoxelDisplayPass(DX12Lib::GraphicsContext& context)
{
	auto& currentBackBuffer = Renderer::GetCurrentBackBuffer();

	context.TransitionResource(currentBackBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, true);

	context.ClearColor(currentBackBuffer, Color::LightSteelBlue().GetPtr(), nullptr);
	context.ClearDepthAndStencil(*Renderer::s_depthStencilBuffer);

	context.SetRenderTargets(1, &currentBackBuffer.GetRTV(), Renderer::s_depthStencilBuffer->GetDSV());

	context.SetViewportAndScissor(Renderer::s_screenViewport, Renderer::s_scissorRect);

	context.SetPipelineState(Renderer::s_PSOs[L"PSO_DISPLAY_VOXEL"].get());

	context.m_commandList->Get()->SetGraphicsRootConstantBufferView((UINT)DisplayVoxelRootParameterSlot::VoxelCommonCBV,
				Renderer::s_graphicsMemory->AllocateConstant(m_cbVoxelCommons).GpuAddress());

	context.m_commandList->Get()->SetGraphicsRootConstantBufferView((UINT)DisplayVoxelRootParameterSlot::CameraCBV,
		m_Scene->GetMainCamera()->GetCameraBuffer().GpuAddress());


	context.m_commandList->Get()->SetGraphicsRootDescriptorTable(
		(UINT)DisplayVoxelRootParameterSlot::VoxelTextureUAV, Renderer::GetVoxelTextureSRV(context));
	

	context.m_commandList->Get()->IASetVertexBuffers(0, 1, &m_vertexBufferView);
	context.m_commandList->Get()->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);

	context.m_commandList->Get()->DrawInstanced((UINT)VoxelTextureDimension.x * (UINT)VoxelTextureDimension.y * (UINT)VoxelTextureDimension.z, 1, 0, 0);
}

void CVGI::ClusteredVoxelGIApp::InitializeVoxelDisplayBuffers()
{
	std::vector<DirectX::XMFLOAT3> vertices;

	for (int x = 0; x < VoxelTextureDimension.x; x++)
	{
		for (int y = 0; y < VoxelTextureDimension.y; y++)
		{
			for (int z = 0; z < VoxelTextureDimension.z; z++)
			{
				vertices.push_back(DirectX::XMFLOAT3(x, y, z));
			}
		}
	}

	UINT vertexStride = sizeof(DirectX::VertexPosition);
	m_vertexBufferResource = Utils::CreateDefaultBuffer(vertices.data(), vertexStride * vertices.size());
	
	m_vertexBufferView.BufferLocation = m_vertexBufferResource->GetGPUVirtualAddress();
	m_vertexBufferView.StrideInBytes = vertexStride;
	m_vertexBufferView.SizeInBytes = vertexStride * vertices.size();
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
	(*voxelizeSceneRootSignature)[(UINT)VoxelizeSceneRootParameterSlot::VoxelTextureUAV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 1);
	(*voxelizeSceneRootSignature)[(UINT)VoxelizeSceneRootParameterSlot::VoxelDataUAV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 2, D3D12_SHADER_VISIBILITY_ALL, 1);
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
	(*displayVoxelRootSignature)[(UINT)DisplayVoxelRootParameterSlot::VoxelTextureUAV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1);
	displayVoxelRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

	return displayVoxelRootSignature;
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::ClusteredVoxelGIApp::BuildVoxelComputeRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> voxelComputeRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)VoxelizeSceneComputeRootParameterSlot::Count, 1);
	voxelComputeRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*voxelComputeRootSignature)[(UINT)VoxelizeSceneComputeRootParameterSlot::VoxelCommonCBV].InitAsConstantBuffer(0);
	(*voxelComputeRootSignature)[(UINT)VoxelizeSceneComputeRootParameterSlot::ObjectCBV].InitAsConstantBuffer(1);
	(*voxelComputeRootSignature)[(UINT)VoxelizeSceneComputeRootParameterSlot::VerticesSRV].InitAsBufferSRV(0, D3D12_SHADER_VISIBILITY_ALL, 1);
	(*voxelComputeRootSignature)[(UINT)VoxelizeSceneComputeRootParameterSlot::IndicesSRV].InitAsBufferSRV(1, D3D12_SHADER_VISIBILITY_ALL, 1);
	(*voxelComputeRootSignature)[(UINT)VoxelizeSceneComputeRootParameterSlot::MaterialSRV].InitAsBufferSRV(2, D3D12_SHADER_VISIBILITY_ALL, 1);
	(*voxelComputeRootSignature)[(UINT)VoxelizeSceneComputeRootParameterSlot::MaterialTextureSRV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, NUM_PBR_TEXTURES);
	(*voxelComputeRootSignature)[(UINT)VoxelizeSceneComputeRootParameterSlot::VoxelTextureUAV].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 1);

	voxelComputeRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

	return voxelComputeRootSignature;
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
	displayVoxelPSO->SetInputLayout(DirectX::VertexPosition::InputLayout.pInputElementDescs, \
		DirectX::VertexPosition::InputLayout.NumElements);
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

std::shared_ptr<DX12Lib::ComputePipelineState> CVGI::ClusteredVoxelGIApp::BuildVoxelComputePso(std::shared_ptr<DX12Lib::RootSignature> voxelRootSig)
{
	std::wstring shaderPath = Utils::ToWstring(SOURCE_DIR) + L"\\Shaders";
	std::wstring computeShaderPath = shaderPath + L"\\VoxelizeScene_CS.hlsl";

	std::shared_ptr<Shader> computeShader = std::make_shared<Shader>(computeShaderPath, "CS", "cs_5_1");
	computeShader->Compile();

	std::shared_ptr<ComputePipelineState> voxelComputePSO = std::make_shared<ComputePipelineState>();
	voxelComputePSO->SetRootSignature(voxelRootSig);
	voxelComputePSO->SetComputeShader(computeShader);
	voxelComputePSO->Finalize();
	voxelComputePSO->Name = L"PSO_VOXELIZE_SCENE_COMPUTE";

	return voxelComputePSO;
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