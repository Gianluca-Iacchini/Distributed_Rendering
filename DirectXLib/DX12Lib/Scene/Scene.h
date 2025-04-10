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

		bool AddFromFile(SceneNode* node, const std::wstring& filename);
		bool AddFromFile(SceneNode* node, const wchar_t* filename);
		bool AddFromFile(SceneNode* node, const char* filename);
		virtual void OnAppStart(GraphicsContext& context) {}
		virtual void Init(GraphicsContext& context);
		virtual void Update(GraphicsContext& context);
		virtual void Render(GraphicsContext& context);
		virtual void OnResize(GraphicsContext& context, int newWidth, int newHeight);
		virtual void OnClose(GraphicsContext& context);

		SceneNode* AddNode();

		SceneNode* GetRootNode() const { return m_rootNode.get(); }

		SceneCamera* GetMainCamera() const { return m_camera; }

		DX12Lib::AABB GetSceneBounds() const { return m_sceneBounds; }

	protected:
		virtual void OnModelChildAdded(SceneNode& node, MeshRenderer& meshRenderer, ModelRenderer& modelRenderer);

	private:
		void TraverseModel(ModelRenderer* model, aiNode* node, SceneNode* parent);


	protected:
		SceneCamera* m_camera = nullptr;
		DX12Lib::AABB m_sceneBounds;
	private:
		NodePtr m_rootNode;

		UINT m_numNodes = 0;


	};
}


