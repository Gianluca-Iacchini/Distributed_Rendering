#pragma once
#include "DX12Lib/Scene/Scene.h"

namespace LI
{
	class LIScene : public DX12Lib::Scene
	{
	public:
		LIScene() : DX12Lib::Scene() {};
		~LIScene() = default;

		virtual void Init(DX12Lib::CommandContext& context) override;
		//virtual void Update(DX12Lib::CommandContext& context) override;
		//virtual void Render(DX12Lib::CommandContext& context) override;
		//virtual void OnResize(DX12Lib::CommandContext& context, int width, int height) override;

		//void TraverseModel(DX12Lib::ModelRenderer* model, aiNode* node, DX12Lib::SceneNode* parent);
	};
}


