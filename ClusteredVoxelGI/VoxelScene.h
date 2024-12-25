#pragma once

#include "DX12Lib/Scene/Scene.h"
#include "VoxelCamera.h"

namespace CVGI
{
	class VoxelScene : public DX12Lib::Scene
	{
	public:
		VoxelScene() = default;
		virtual ~VoxelScene() = default;

		virtual void Init(DX12Lib::CommandContext& context) override;
		virtual void Update(DX12Lib::CommandContext& context) override;
		virtual void Render(DX12Lib::CommandContext& context) override;
		virtual void OnResize(DX12Lib::CommandContext& context, int newWidth, int newHeight) override;
		virtual void OnClose(DX12Lib::CommandContext& context) override;

		VoxelCamera* GetVoxelCamera() { return m_voxelCamera; }
		DX12Lib::LightComponent* GetMainLight() { return m_mainLight; }

	protected:
		virtual void OnModelChildAdded(DX12Lib::SceneNode& modelNode, DX12Lib::MeshRenderer& meshRenderer, DX12Lib::ModelRenderer& modelRenderer) override;

	public:
		DirectX::XMFLOAT3 VoxelTextureDimensions = DirectX::XMFLOAT3(128.0f, 128.0f, 128.0f);
		DirectX::XMFLOAT3 VoxelCellSize = DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f);
		DX12Lib::LightComponent* m_mainLight = nullptr;

	private:
		DX12Lib::ColorBuffer m_voxelTexture;
		VoxelCamera* m_voxelCamera = nullptr;
	};
}