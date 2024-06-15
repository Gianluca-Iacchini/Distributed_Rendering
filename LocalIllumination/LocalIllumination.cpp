#define STREAMING 0

#include <DX12Lib/pch.h>

#include "DX12Lib/Commons/D3DApp.h"
#include "LIScene.h"
#include "DX12Lib/Models/ModelRenderer.h"


using namespace DirectX;
using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;


class LocalIlluminationApp : public D3DApp
{
private:
	bool m_usePBRMaterials = true;

public:
	LocalIlluminationApp(HINSTANCE hInstance, Scene* scene = nullptr) : D3DApp(hInstance, scene) {};
	LocalIlluminationApp(const LocalIlluminationApp& rhs) = delete;
	LocalIlluminationApp& operator=(const LocalIlluminationApp& rhs) = delete;
	~LocalIlluminationApp() { 

		FlushCommandQueue();
	};

	virtual void Initialize(CommandContext& context) override
	{

		std::string sourcePath = std::string(SOURCE_DIR);

		if (m_usePBRMaterials)
			sourcePath += std::string("\\Models\\PBR\\sponza2.gltf");
		else
			sourcePath += std::string("\\Models\\sponza_nobanner.obj");


		bool loaded = this->m_Scene->AddFromFile(sourcePath.c_str());

		assert(loaded && "Model not loaded");



		s_mouse->SetMode(Mouse::MODE_RELATIVE);

		if (!m_usePBRMaterials)
		{
			auto rootNode = m_Scene->GetRootNode();

			rootNode->SetScale(0.01f, 0.01f, 0.01f);
		}

		this->m_Scene->Init(context);
	}


	virtual void Update(CommandContext& context) override
	{
		D3DApp::Update(context);

	}

	virtual void Draw(CommandContext& context) override
	{



		Renderer::SetUpRenderFrame(context);


		this->m_Scene->Render(context);



		Renderer::RenderLayers(context);
		
		LI::LIScene* scene = dynamic_cast<LI::LIScene*>(this->m_Scene.get());
		
		if (scene != nullptr)
			scene->StreamScene(context);
	}

	virtual void OnClose(CommandContext& context) override
	{
		D3DApp::OnClose(context);
	}
};


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance, PSTR cmdLine, int showCmd)
{
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
	try
	{
		LocalIlluminationApp app(hInstance, new LI::LIScene(STREAMING));
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