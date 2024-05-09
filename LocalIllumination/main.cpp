#include <DX12Lib/pch.h>
#include "DX12Lib/Helpers.h"
#include "DX12Lib/D3DApp.h"
#include "DX12Lib/CIDXGIFactory.h"
#include "Keyboard.h"
#include "DX12Lib/Shader.h"
#include "Dx12Lib/PipelineState.h"
#include "FrameResource.h"
#include "DX12Lib/Fence.h"
#include "DX12Lib/CommandQueue.h"
#include "DX12Lib/CommandList.h"
#include "DX12Lib/CommandAllocator.h"
#include "DX12Lib/Swapchain.h"
#include "DX12Lib/DepthBuffer.h"
#include "DX12Lib/CommandContext.h"
#include "DX12Lib/RootSignature.h"
#include "DX12Lib/CommandContext.h"
#include "DX12Lib/Texture.h"
#include "DX12Lib/DescriptorHeap.h"
#include "DX12Lib/SamplerDesc.h"
#include "GraphicsMemory.h"
#include "GeometricPrimitive.h"
#include "DX12Lib/CommonConstants.h"
#include "DX12Lib/Camera.h"
#include "Mouse.h"
#include "ResourceUploadBatch.h"
#include "DX12Lib/Model.h"
#include "DX12Lib/Mesh.h"

using namespace DirectX;
using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;

struct Vertex
{
	Vertex(XMFLOAT3 p, XMFLOAT3 n, XMFLOAT2 tc) : Pos(p), TexCoord(tc), Normal(n) {}

	XMFLOAT3 Pos;
	XMFLOAT3 Normal;
	XMFLOAT2 TexCoord;
};


class AppTest : public D3DApp
{
	Keyboard keyboard;
	DirectX::Keyboard::KeyboardStateTracker tracker;

	std::shared_ptr<RootSignature> m_rootSignature;
	PipelineState m_pipelineState;

	std::unordered_map<std::string, std::shared_ptr<Shader>> mp_shaders;

	std::vector<D3D12_INPUT_ELEMENT_DESC> m_inputLayout;

	UINT64 frameFences[3] = { 0, 0, 0 };


	CostantBufferCommons m_costantBufferCommons;
	ConstantBufferObject m_costantBufferObject;

	Camera camera;

	std::unique_ptr<DirectX::GeometricPrimitive> m_shape;

	Model m_model;

	float cameraSpeed = 100.0f;

	float m_theta = 1.25f * XM_PI;
	float m_phi = XM_PIDIV4;
	XMFLOAT2 m_modifier = XMFLOAT2(0.85f, 0.85f);

public:
	AppTest(HINSTANCE hInstance) : D3DApp(hInstance) {};
	AppTest(const AppTest& rhs) = delete;
	AppTest& operator=(const AppTest& rhs) = delete;
	~AppTest() { 

		FlushCommandQueue();
	};

	void BuildShadersAndInputLayout()
	{
		std::wstring srcDir = Utils::ToWstring(SOURCE_DIR);
		std::wstring VSshaderFile = srcDir + L"\\Shaders\\Basic_VS.hlsl";
		std::wstring PSshaderFile = srcDir + L"\\Shaders\\Basic_PS.hlsl";

		std::shared_ptr<Shader> vertexShader = std::make_shared<Shader>(VSshaderFile, "VS", "vs_5_1");
		std::shared_ptr<Shader> pixelShader = std::make_shared<Shader>(PSshaderFile, "PS", "ps_5_1");

		vertexShader->Compile();
		pixelShader->Compile();

		mp_shaders["basicVS"] = std::move(vertexShader);
		mp_shaders["basicPS"] = std::move(pixelShader);

		m_inputLayout =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			{ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
			{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
		};
	}


	void BuildEmptyRootSignature()
	{
		SamplerDesc DefaultSamplerDesc;
		DefaultSamplerDesc.MaxAnisotropy = 8;

		m_rootSignature = std::make_shared<RootSignature>(4, 1);
		(*m_rootSignature)[0].InitAsConstantBuffer(0);
		(*m_rootSignature)[1].InitAsConstantBuffer(1);
		(*m_rootSignature)[2].InitAsConstantBuffer(2);
		(*m_rootSignature)[3].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, (UINT)MaterialTextureType::NUM_TEXTURE_TYPES);
		m_rootSignature->InitStaticSampler(0, DefaultSamplerDesc);
		m_rootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
	
	}

	void BuildPSO()
	{
		m_pipelineState = PipelineState();
		m_pipelineState.InitializeDefaultStates();
		m_pipelineState.SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
		m_pipelineState.SetShader(mp_shaders["basicVS"], ShaderType::Vertex);
		m_pipelineState.SetShader(mp_shaders["basicPS"], ShaderType::Pixel);
		m_pipelineState.SetInputLayout(VertexPositionNormalTexture::InputLayout.pInputElementDescs, VertexPositionNormalTexture::InputLayout.NumElements);
		m_pipelineState.SetRootSignature(m_rootSignature);
		m_pipelineState.SetRenderTargetFormat(mBackBufferFormat, mDepthStencilFormat, 1, 0);

		m_pipelineState.Finalize();
	}
	virtual bool Initialize() override
	{
		if (!D3DApp::Initialize())
			return false;

		CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);


		BuildEmptyRootSignature();
		BuildShadersAndInputLayout();
		BuildPSO();

		std::string sourcePath = std::string(SOURCE_DIR) + std::string("\\Models\\sponza_nobanner.obj");

		bool loaded = m_model.LoadFromFile(sourcePath.c_str());

		assert(loaded && "Model not loaded");

		camera.SetPosition(0.0f, 250.0f, 0.0f);

		auto cameraPos = camera.GetPosition3f();
		auto cameraLookAt = XMFLOAT3(cameraPos.x + 1.0f, cameraPos.y, cameraPos.z);

		camera.LookAt(cameraPos, XMFLOAT3(cameraLookAt), XMFLOAT3(0.0f, 1.0f, 0.0f));

		context->Finish(true);

		s_mouse->SetMode(Mouse::MODE_RELATIVE);

		return true;
	}

	void UpdateCommonConstants(const GameTime& gt)
	{
		XMMATRIX view = camera.GetView();
		XMMATRIX invView = XMMatrixInverse(nullptr, view);
		XMMATRIX proj = camera.GetProjection();
		XMMATRIX invProj = XMMatrixInverse(nullptr, proj);
		XMMATRIX viewProj = XMMatrixMultiply(view, proj);
		XMMATRIX invViewProj = XMMatrixInverse(nullptr, viewProj);



		XMStoreFloat4x4(&m_costantBufferCommons.view, DirectX::XMMatrixTranspose(view));
		XMStoreFloat4x4(&m_costantBufferCommons.invView, DirectX::XMMatrixTranspose(invView));
		XMStoreFloat4x4(&m_costantBufferCommons.projection, DirectX::XMMatrixTranspose(proj));
		XMStoreFloat4x4(&m_costantBufferCommons.invProjection, DirectX::XMMatrixTranspose(invProj));
		XMStoreFloat4x4(&m_costantBufferCommons.viewProjection, DirectX::XMMatrixTranspose(viewProj));
		XMStoreFloat4x4(&m_costantBufferCommons.invViewProjection, DirectX::XMMatrixTranspose(invViewProj));
		m_costantBufferCommons.eyePosition = camera.GetPosition3f();
		m_costantBufferCommons.nearPlane = camera.GetNearZ();
		m_costantBufferCommons.renderTargetSize = XMFLOAT2((float)mClientWidth, (float)mClientHeight);
		m_costantBufferCommons.invRenderTargetSize = XMFLOAT2(1.0f / mClientWidth, 1.0f / mClientHeight);
		m_costantBufferCommons.farPlane = camera.GetFarZ();
		m_costantBufferCommons.totalTime = gt.TotalTime();
		m_costantBufferCommons.deltaTime = gt.DeltaTime();

		Light dirLight;
		XMVECTOR lightDir = MathHelper::SphericalToCartesian(1.0f, m_theta, m_phi); //XMFLOAT3(-0.57735f, -0.57735f, 0.57735f);
		
		XMStoreFloat3(&dirLight.Direction, lightDir);
		dirLight.Color = XMFLOAT3(1.3f, 1.3f, 1.3f);

		m_costantBufferCommons.lights[0] = dirLight;
	}

	void MoveCamera()
	{
		auto state = s_mouse->GetState();
		if (state.positionMode == Mouse::MODE_RELATIVE)
		{

			float dx = XMConvertToRadians(0.25f * static_cast<float>(state.x));
			float dy = XMConvertToRadians(0.25f * static_cast<float>(state.y));

			camera.Pitch(dy);
			camera.Yaw(dx);
		}
	}

	void UpdateKeyboard(const GameTime& gt)
	{
		auto kbState = keyboard.GetState();
		tracker.Update(kbState);

		if (tracker.IsKeyPressed(DirectX::Keyboard::Keys::Escape))
		{
			PostQuitMessage(0);
		}
		if (kbState.W)
		{
			camera.Walk(cameraSpeed * gt.DeltaTime());
		}
		if (kbState.S)
		{
			camera.Walk(-cameraSpeed * gt.DeltaTime());
		}
		if (kbState.A)
		{
			camera.Strafe(-cameraSpeed * gt.DeltaTime());
		}
		if (kbState.D)
		{
			camera.Strafe(cameraSpeed * gt.DeltaTime());
		}
		if (kbState.E)
		{
			camera.Lift(cameraSpeed * gt.DeltaTime());
		}
		if (kbState.Q)
		{
			camera.Lift(-cameraSpeed * gt.DeltaTime());
		}
	}

	virtual void Update(const GameTime& gt) override
	{
		UINT64 currentFrame = frameFences[m_swapchain->CurrentBufferIndex];
		if (currentFrame != 0)
		{
			s_commandQueueManager->GetGraphicsQueue().WaitForFence(currentFrame);
		}

		// Update sun orientation
		m_theta +=  m_modifier.x * gt.DeltaTime();
		m_phi += m_modifier.y * gt.DeltaTime();

		if (m_phi > XM_PI || m_phi < -XM_PI)
			m_modifier.y = -m_modifier.y;
		
		if (m_theta > XM_PI || m_theta < -XM_PI)
			m_modifier.x = -m_modifier.x;




		UpdateKeyboard(gt);
		MoveCamera();
		camera.UpdateViewMatrix();
		UpdateCommonConstants(gt);
	}

	virtual void Draw(const GameTime& gt) override
	{
		CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);

		context->m_commandList->GetComPtr()->RSSetViewports(1, &mScreenViewport);
		context->m_commandList->GetComPtr()->RSSetScissorRects(1, &mScissorRect);

		context->TransitionResource(CurrentBackBuffer(), D3D12_RESOURCE_STATE_RENDER_TARGET, true);

		float clearDepth = m_depthStencilBuffer->GetClearDepth();
		float clearStencil = m_depthStencilBuffer->GetClearStencil();

		context->m_commandList->GetComPtr()->ClearRenderTargetView(CurrentBackBufferView(), DirectX::Colors::LightSteelBlue, 0, nullptr);
		context->m_commandList->GetComPtr()->ClearDepthStencilView(DepthStencilView(), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, clearDepth, clearStencil, 0, nullptr);

		context->m_commandList->GetComPtr()->OMSetRenderTargets(1, &CurrentBackBufferView(), true, &DepthStencilView());

		context->m_commandList->GetComPtr()->SetGraphicsRootSignature(m_rootSignature->Get());

		context->m_commandList->GetComPtr()->SetPipelineState(m_pipelineState.Get());

		auto commonRes = s_graphicsMemory->AllocateConstant(m_costantBufferCommons);
		auto objectRes = s_graphicsMemory->AllocateConstant(m_costantBufferObject);

		context->m_commandList->GetComPtr()->SetGraphicsRootConstantBufferView(0, commonRes.GpuAddress());
		context->m_commandList->GetComPtr()->SetGraphicsRootConstantBufferView(1, objectRes.GpuAddress());



		m_model.Draw(*context);


		context->TransitionResource(CurrentBackBuffer(), D3D12_RESOURCE_STATE_PRESENT, true);

		frameFences[m_swapchain->CurrentBufferIndex] = context->Finish();

		HRESULT hr = m_swapchain->GetComPointer()->Present(0, 0);

		if (FAILED(hr))
		{
			hr = s_device->GetComPtr()->GetDeviceRemovedReason();
			if (FAILED(hr))
			{
				DeviceRemovedHandler();
				ThrowIfFailed(hr);
			}
		}

		CommandContext::CommitGraphicsResources(D3D12_COMMAND_LIST_TYPE_DIRECT);
		m_swapchain->CurrentBufferIndex = (m_swapchain->CurrentBufferIndex + 1) % m_swapchain->BufferCount;
	}

	virtual void OnResize() override
	{
		D3DApp::OnResize();

		camera.SetLens(0.25f * DirectX::XM_PI, AspectRatio(), 1.0f, 10000.0f);
	}
};


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance, PSTR cmdLine, int showCmd)
{
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
	try
	{
		AppTest app(hInstance);
		if (!app.Initialize())
			return 0;

		return app.Run();
	}
	catch (DxException& e)
	{
		MessageBox(nullptr, e.ToString().c_str(), L"HR Failed", MB_OK);
		return 0;
	}
}