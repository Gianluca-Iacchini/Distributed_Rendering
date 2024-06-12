#pragma once
#include "SceneNode.h"
#include "DX12Lib/Models/Model.h"
#include "DX12Lib/Commons/GameTime.h"


namespace DX12Lib
{
	class ModelRenderer;
	class SceneCamera;
	class D3DApp;

	class Scene
	{
		friend class SceneNode;
	public:
		Scene();
		~Scene();

		bool AddFromFile(const std::wstring& filename);
		bool AddFromFile(const wchar_t* filename);
		bool AddFromFile(const char* filename);
		virtual void Init(CommandContext& context);
		virtual void Update(CommandContext& context);
		virtual void Render(CommandContext& context);
		virtual void OnResize(CommandContext& context, int newWidth, int newHeight);

		SceneNode* AddNode();

		void SetNetworkData(const char* data, size_t size);
		const std::vector<char>& GetNetworkData() const { return m_inputData; }

	private:
		void TraverseModel(ModelRenderer* model, aiNode* node, SceneNode* parent);

	protected:
		SceneCamera* m_camera = nullptr;

	private:
		NodePtr m_rootNode;

		UINT m_numNodes = 0;



		std::vector<char> m_inputData;
	};
}


