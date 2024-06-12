//#include <DX12Lib/pch.h>
//
//#include "DX12Lib/Commons/D3DApp.h"
//#include "Keyboard.h"
//
//#include "Dx12Lib/DXWrapper/PipelineState.h"
//#include "FrameResource.h"
//#include "DX12Lib/DXWrapper/Swapchain.h"
//#include "GraphicsMemory.h"
//#include "GeometricPrimitive.h"
//#include "DX12Lib/Commons/Camera.h"
//#include "Mouse.h"
//#include "ResourceUploadBatch.h"
//#include "DX12Lib/Scene/Scene.h"
//#include "DX12Lib/DXWrapper/Swapchain.h"
//
//
//
//using namespace DirectX;
//using namespace Microsoft::WRL;
//using namespace Graphics;
//using namespace DX12Lib;
//
//#define USE_PBR 1
//
//
//class AppTest : public D3DApp
//{
//
//	CostantBufferCommons m_costantBufferCommons;
//	std::unique_ptr<Scene> m_scene;
//
//public:
//	AppTest(HINSTANCE hInstance) : D3DApp(hInstance) {};
//	AppTest(const AppTest& rhs) = delete;
//	AppTest& operator=(const AppTest& rhs) = delete;
//	~AppTest() { 
//
//		FlushCommandQueue();
//	};
//
//	virtual bool Initialize() override
//	{
//		if (!D3DApp::Initialize())
//			return false;
//
//		CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);
//
//
//
//#if USE_PBR
//		std::string sourcePath = std::string(SOURCE_DIR) + std::string("\\Models\\PBR\\sponza2.gltf");
//#else
//		std::string sourcePath = std::string(SOURCE_DIR) + std::string("\\Models\\sponza_nobanner.obj");
//#endif
//		m_scene = std::make_unique<Scene>();
//
//		bool loaded = m_scene->AddFromFile(sourcePath.c_str());
//
//		assert(loaded && "Model not loaded");
//
//		m_scene->Init(*context);
//
//
//		context->Finish(true);
//
//		s_mouse->SetMode(Mouse::MODE_RELATIVE);
//
//		return true;
//	}
//
//	void UpdateCommonConstants()
//	{
//
//		m_costantBufferCommons.renderTargetSize = XMFLOAT2((float)Renderer::s_clientWidth, (float)Renderer::s_clientHeight);
//		m_costantBufferCommons.invRenderTargetSize = XMFLOAT2(1.0f / Renderer::s_clientWidth, 1.0f / Renderer::s_clientHeight);
//		m_costantBufferCommons.totalTime = GameTime::GetTotalTime();
//		m_costantBufferCommons.deltaTime = GameTime::GetDeltaTime();
//	}
//
//	virtual void Update() override
//	{
//		Renderer::WaitForSwapchainBuffers();
//
//		CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);
//
//		auto kbState = s_keyboard->GetState();
//		s_kbTracker->Update(kbState);
//
//		UpdateCommonConstants();
//
//		m_scene->Update(*context);
//
//		context->Finish(true);
//	}
//
//	virtual void Draw() override
//	{
//
//		CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);
//
//		auto commonRes = Renderer::s_graphicsMemory->AllocateConstant(m_costantBufferCommons);
//
//		Renderer::SetUpRenderFrame(context);
//		
//		context->m_commandList->GetComPtr()->SetGraphicsRootConstantBufferView(
//			(UINT)Renderer::RootSignatureSlot::CommonCBV, commonRes.GpuAddress()
//		);
//
//		m_scene->Render(*context);
//
//		Renderer::RenderLayers(context);
//		
//		auto& backBuffer = Renderer::GetCurrentBackBuffer();
//
//		context->TransitionResource(backBuffer, D3D12_RESOURCE_STATE_PRESENT, true);
//
//		auto fenceVal = context->Finish(true);
//
//		Renderer::Present(fenceVal);
//
//	}
//
//	virtual void OnResize(CommandContext& context) override
//	{
//		D3DApp::OnResize(context);
//
//		if (m_scene != nullptr)
//			m_scene->OnResize(context);
//	}
//
//	virtual void OnClose() override
//	{
//
//	}
//};
//
//
//int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance, PSTR cmdLine, int showCmd)
//{
//#if defined(DEBUG) | defined(_DEBUG)
//	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
//#endif
//	try
//	{
//		AppTest app(hInstance);
//		if (!app.Initialize())
//			return 0;
//
//		return app.Run();
//	}
//	catch (DxException& e)
//	{
//		MessageBox(nullptr, e.ToString().c_str(), L"HR Failed", MB_OK);
//		return 0;
//	}
//}