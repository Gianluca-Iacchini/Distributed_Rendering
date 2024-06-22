#pragma once

#include "DX12Lib/Scene/Scene.h"

namespace CVGI
{
	enum class VoxelRootParameter
	{
		CommonCBV = 0,
		ObjectCBV = 1,
		VoxelDataCBV,
		LightSRV,
		MaterialSRV,
		VoxelTextureSRV,
		MaterialTextureSRV,
		Count
	};



	class VoxelScene : public DX12Lib::Scene
	{
	public:
		VoxelScene() = default;
		virtual ~VoxelScene() = default;

		virtual void OnAppStart(DX12Lib::CommandContext& context) override;
		virtual void Init(DX12Lib::CommandContext& context) override;
		virtual void Update(DX12Lib::CommandContext& context) override;
		virtual void Render(DX12Lib::CommandContext& context) override;
		virtual void OnResize(DX12Lib::CommandContext& context, int newWidth, int newHeight) override;
		virtual void OnClose(DX12Lib::CommandContext& context) override;

	protected:
		virtual void OnModelChildAdded(DX12Lib::SceneNode& modelNode, DX12Lib::MeshRenderer& meshRenderer, DX12Lib::ModelRenderer& modelRenderer) override;
	

	private:
		std::shared_ptr<DX12Lib::RootSignature> BuildVoxelRootSignature();
		std::shared_ptr<DX12Lib::PipelineState> BuildVoxelPSO();
	};
}