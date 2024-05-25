#pragma once
#include "SceneNode.h"
#include "DX12Lib/Models/Model.h"
#include "DX12Lib/Commons/GameTime.h"

namespace DX12Lib
{
	class ModelRenderer;
	class SceneCamera;

	class Scene
	{
		friend class SceneNode;
	public:
		Scene(GameTime& time);
		~Scene();

		bool AddFromFile(const std::wstring& filename);
		bool AddFromFile(const wchar_t* filename);
		bool AddFromFile(const char* filename);
		void Init(CommandContext& context);
		void Update(CommandContext& context);
		void Render(CommandContext& context);
		void OnResize(CommandContext& context);
		void Draw(ID3D12GraphicsCommandList* cmdList);

		void TraverseModel(ModelRenderer* model, aiNode* node, SceneNode* parent);

		inline const GameTime& Time() const { return m_time; }

	private:
		NodePtr m_rootNode;
		GameTime& m_time;

		UINT m_numNodes = 0;

		SceneCamera* m_camera = nullptr;
	};
}


