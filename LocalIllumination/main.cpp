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
#include "DX12Lib/Scene/Scene.h"
#include "DX12Lib/DXWrapper/Swapchain.h"

using namespace DirectX;
using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;

#define USE_PBR 1


class AppTest : public D3DApp
{
	UINT64 frameFences[3] = { 0, 0, 0 };


	CostantBufferCommons m_costantBufferCommons;
	ConstantBufferObject m_costantBufferObject;

	std::unique_ptr<DirectX::GeometricPrimitive> m_shape;

	std::unique_ptr<Scene> m_scene;

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
		std::string sourcePath = std::string(SOURCE_DIR) + std::string("\\Models\\PBR\\sponza2.gltf");
#else
		std::string sourcePath = std::string(SOURCE_DIR) + std::string("\\Models\\sponza_nobanner.obj");
#endif
		m_scene = std::make_unique<Scene>(this->m_Time);

		bool loaded = m_scene->AddFromFile(sourcePath.c_str());

		assert(loaded && "Model not loaded");

		m_scene->Init(*context);

		context->Finish(true);

		s_mouse->SetMode(Mouse::MODE_RELATIVE);

		return true;
	}

	void UpdateCommonConstants(const GameTime& gt)
	{

		m_costantBufferCommons.renderTargetSize = XMFLOAT2((float)mClientWidth, (float)mClientHeight);
		m_costantBufferCommons.invRenderTargetSize = XMFLOAT2(1.0f / mClientWidth, 1.0f / mClientHeight);
		m_costantBufferCommons.totalTime = gt.TotalTime();
		m_costantBufferCommons.deltaTime = gt.DeltaTime();

		//ConstantBufferLight dirLight;
		//XMVECTOR lightDir = MathHelper::SphericalToCartesian(3.0, m_theta, m_phi); //XMFLOAT3(-0.57735f, -0.57735f, 0.57735f);
		//
		//XMStoreFloat3(&dirLight.Direction, lightDir);
		//dirLight.Color = XMFLOAT3(0.6, 0.6, 0.6);

		//m_costantBufferCommons.light = dirLight;
	}

	virtual void Update(const GameTime& gt) override
	{
		Renderer::WaitForSwapchainBuffers();

		// Update sun orientation
		m_theta +=  m_modifier.x * gt.DeltaTime();
		m_phi += m_modifier.y * gt.DeltaTime();


		CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);

		auto kbState = s_keyboard->GetState();
		s_kbTracker->Update(kbState);

		UpdateCommonConstants(gt);
		m_scene->Update(*context);

		context->Finish(true);
	}

	virtual void Draw(const GameTime& gt) override
	{
		CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);

		auto commonRes = Renderer::s_graphicsMemory->AllocateConstant(m_costantBufferCommons);

		Renderer::SetUpRenderFrame(context);
		
		context->m_commandList->GetComPtr()->SetGraphicsRootConstantBufferView(
			(UINT)Renderer::RootSignatureSlot::CommonCBV, commonRes.GpuAddress()
		);

		m_scene->Render(*context);

		Renderer::RenderLayers(context);

		auto fenceVal = context->Finish(true);

		Renderer::Present(fenceVal);
	}

	virtual void OnResize() override
	{
		D3DApp::OnResize();
		
		CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);

		if (m_scene != nullptr)
			m_scene->OnResize(*context);

		context->Finish(true);
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