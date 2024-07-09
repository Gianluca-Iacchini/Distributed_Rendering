#pragma once
#include <DX12Lib/Commons/D3DApp.h>
#include <memory>
#include <DirectXMath.h>
#include "DX12Lib/DXWrapper/RootSignature.h"
#include "DX12Lib/DXWrapper/PipelineState.h"
#include "CVGIDataTypes.h"
#include "DX12Lib/DXWrapper/GPUBuffer.h"
#include "DX12Lib/DXWrapper/DescriptorHeap.h"

namespace CVGI
{
	class VoxelCamera;





	class ClusteredVoxelGIApp : public DX12Lib::D3DApp
	{
	public:

		ClusteredVoxelGIApp(HINSTANCE hInstance, DX12Lib::Scene* scene)
			: D3DApp(hInstance, scene)
		{
		}

		~ClusteredVoxelGIApp() = default;

		virtual void Initialize(DX12Lib::GraphicsContext& commandContext) override;
		virtual void Draw(DX12Lib::GraphicsContext& commandContext) override;

		void VoxelPass(DX12Lib::GraphicsContext& context, VoxelCamera* voxelCamera);
		void VoxelDisplayPass(DX12Lib::GraphicsContext& context);
		void VoxelComputePass();
	private:

		void InitializeVoxelDisplayBuffers();


		std::shared_ptr<DX12Lib::RootSignature> BuildVoxelizeSceneRootSignature();
		std::shared_ptr<DX12Lib::RootSignature> BuildVoxelDisplayRootSignature();
		std::shared_ptr<DX12Lib::RootSignature> BuildVoxelComputeRootSignature();


		std::shared_ptr<DX12Lib::GraphicsPipelineState> BuildVoxelizeScenePso(std::shared_ptr<DX12Lib::RootSignature> voxelRootSig);
		std::shared_ptr<DX12Lib::GraphicsPipelineState> BuildVoxelDisplayPso(std::shared_ptr<DX12Lib::RootSignature> voxelRootSig);
		std::shared_ptr<DX12Lib::ComputePipelineState> BuildVoxelComputePso(std::shared_ptr<DX12Lib::RootSignature> voxelRootSig);

	public:
		const DirectX::XMFLOAT3 VoxelTextureDimension = DirectX::XMFLOAT3(196.0f, 196.0f, 196.0f);

	private:
		ConstantBufferVoxelCommons m_cbVoxelCommons;
		DX12Lib::ColorBuffer m_voxelTexture;
		DX12Lib::StructuredBuffer m_voxelDataBuffer;

		DX12Lib::DescriptorHandle m_voxelDataUAVStart;

		Microsoft::WRL::ComPtr<ID3D12Resource> m_vertexBufferResource;

		D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;

		D3D12_VIEWPORT m_voxelScreenViewport;
		D3D12_RECT m_voxelScissorRect;
	};
}