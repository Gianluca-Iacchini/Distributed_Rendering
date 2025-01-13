#pragma once
#include "SceneNode.h"
#include "DX12Lib/Models/Model.h"
#include "DX12Lib/Commons/GameTime.h"


namespace DX12Lib
{
	class ModelRenderer;
	class D3DApp;
	class SceneCamera;

	class Scene
	{
		friend class SceneNode;
	public:
		Scene();
		virtual ~Scene();

		bool AddFromFile(const std::wstring& filename);
		bool AddFromFile(const wchar_t* filename);
		bool AddFromFile(const char* filename);
		virtual void OnAppStart(GraphicsContext& context) {}
		virtual void Init(GraphicsContext& context);
		virtual void Update(GraphicsContext& context);
		virtual void Render(GraphicsContext& context);
		virtual void OnResize(GraphicsContext& context, int newWidth, int newHeight);
		virtual void OnClose(GraphicsContext& context);

		SceneNode* AddNode();

		SceneNode* GetRootNode() const { return m_rootNode.get(); }

		SceneCamera* GetMainCamera() const { return m_camera; }

	protected:
		virtual void OnModelChildAdded(SceneNode& node, MeshRenderer& meshRenderer, ModelRenderer& modelRenderer);

	private:
		void TraverseModel(ModelRenderer* model, aiNode* node, SceneNode* parent);


	protected:
		SceneCamera* m_camera = nullptr;

	private:
		NodePtr m_rootNode;

		UINT m_numNodes = 0;
	};
}


