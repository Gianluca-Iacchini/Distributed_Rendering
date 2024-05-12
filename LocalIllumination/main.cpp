#include <DX12Lib/pch.h>

#include "DX12Lib/Commons/D3DApp.h"
#include "Keyboard.h"

#include "Dx12Lib/DXWrapper/PipelineState.h"
#include "FrameResource.h"
#include "DX12Lib/DXWrapper/Swapchain.h"
#include "GraphicsMemory.h"
#include "GeometricPrimitive.h"
#include "DX12Lib/Commons/Camera.h"
#include "Mouse.h"
#include "ResourceUploadBatch.h"
#include "DX12Lib/Models/Model.h"
#include "DX12Lib/Models/Mesh.h"

using namespace DirectX;
using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;

#define USE_PBR 1

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


	UINT64 frameFences[3] = { 0, 0, 0 };


	CostantBufferCommons m_costantBufferCommons;
	ConstantBufferObject m_costantBufferObject;

	Camera camera;

	std::unique_ptr<DirectX::GeometricPrimitive> m_shape;

	Model m_model;

	float cameraSpeed = 100.0f;

	float m_theta = 1.25f * XM_PI;
	float m_phi = XM_PIDIV4;
	XMFLOAT2 m_modifier = XMFLOAT2(0.45f, 0.45f);

	std::shared_ptr<RootSignature> m_rootSignature;
	std::shared_ptr<PipelineState> m_pipelineState;

public:
	AppTest(HINSTANCE hInstance) : D3DApp(hInstance) {};
	AppTest(const AppTest& rhs) = delete;
	AppTest& operator=(const AppTest& rhs) = delete;
	~AppTest() { 

		FlushCommandQueue();
	};

	virtual bool Initialize() override
	{
		if (!D3DApp::Initialize())
			return false;

		CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);



#if USE_PBR
		m_pipelineState = s_PSOs[L"PBRPSO"];
		m_rootSignature = m_pipelineState->GetRootSignature();
		std::string sourcePath = std::string(SOURCE_DIR) + std::string("\\Models\\PBR\\sponza2.gltf");
#else
		m_pipelineState = s_PSOs[L"opaquePSO"];
		m_rootSignature = m_pipelineState->GetRootSignature();
		std::string sourcePath = std::string(SOURCE_DIR) + std::string("\\Models\\sponza_nobanner.obj");
#endif

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
		XMVECTOR lightDir = MathHelper::SphericalToCartesian(3.0, m_theta, m_phi); //XMFLOAT3(-0.57735f, -0.57735f, 0.57735f);
		
		XMStoreFloat3(&dirLight.Direction, lightDir);
		dirLight.Color = XMFLOAT3(0.6, 0.6, 0.6);

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

		context->m_commandList->GetComPtr()->SetPipelineState(m_pipelineState->Get());

		auto commonRes = s_graphicsMemory->AllocateConstant(m_costantBufferCommons);
		auto objectRes = s_graphicsMemory->AllocateConstant(m_costantBufferObject);

		context->m_commandList->GetComPtr()->SetGraphicsRootConstantBufferView(0, commonRes.GpuAddress());
		context->m_commandList->GetComPtr()->SetGraphicsRootConstantBufferView(1, objectRes.GpuAddress());

		ID3D12DescriptorHeap* heaps[] = { s_textureHeap->Get() };
		context->m_commandList->Get()->SetDescriptorHeaps(1, heaps);

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