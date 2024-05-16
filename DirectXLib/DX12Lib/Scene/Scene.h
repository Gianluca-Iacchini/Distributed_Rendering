#pragma once
#include "SceneNode.h"
#include "DX12Lib/Models/Model.h"

namespace DX12Lib
{
	class ModelRenderer;

	class Scene
	{
		friend class SceneNode;
	public:
		Scene();
		~Scene();

		bool AddFromFile(const std::wstring& filename);
		bool AddFromFile(const wchar_t* filename);
		bool AddFromFile(const char* filename);
		void Update(CommandContext* context);
		void Render(CommandContext* context);
		void Release();
		void Draw(ID3D12GraphicsCommandList* cmdList);

		void Traverse(ModelRenderer* model, aiNode* node, SceneNode* parent);

	private:
		NodePtr m_rootNode;

		UINT m_numNodes = 0;

		std::vector<std::unique_ptr<Model>> m_sceneModels;
	};
}


