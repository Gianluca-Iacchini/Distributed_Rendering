#pragma once
#include <DX12Lib/Commons/D3DApp.h>
#include <memory>
#include <DirectXMath.h>
#include "DX12Lib/DXWrapper/RootSignature.h"
#include "DX12Lib/DXWrapper/PipelineState.h"
#include "CVGIDataTypes.h"
#include "DX12Lib/DXWrapper/GPUBuffer.h"
#include "DX12Lib/DXWrapper/DescriptorHeap.h"
#include "VoxelBufferManager.h"
#include "GraphicsMemory.h"

namespace CVGI
{
	class VoxelCamera;





	class ClusteredVoxelGIApp : public DX12Lib::D3DApp
	{
	public:

		ClusteredVoxelGIApp(HINSTANCE hInstance, DX12Lib::Scene* scene)
			: D3DApp(hInstance, scene), m_vertexBuffer(DX12Lib::TypedBuffer(DXGI_FORMAT_R32_UINT))
		{

		}

		~ClusteredVoxelGIApp() = default;

		virtual void Initialize(DX12Lib::GraphicsContext& commandContext) override;
		virtual void Update(DX12Lib::GraphicsContext& commandContext) override;
		virtual void Draw(DX12Lib::GraphicsContext& commandContext) override;

		void VoxelPass(DX12Lib::GraphicsContext& context, VoxelCamera* voxelCamera);
		void VoxelDisplayPass(DX12Lib::GraphicsContext& context);

	private:

		std::shared_ptr<DX12Lib::RootSignature> BuildVoxelizeSceneRootSignature();
		std::shared_ptr<DX12Lib::RootSignature> BuildVoxelDisplayRootSignature();



		std::shared_ptr<DX12Lib::GraphicsPipelineState> BuildVoxelizeScenePso(std::shared_ptr<DX12Lib::RootSignature> voxelRootSig);
		std::shared_ptr<DX12Lib::GraphicsPipelineState> BuildVoxelDisplayPso(std::shared_ptr<DX12Lib::RootSignature> voxelRootSig);

	public:
		const DirectX::XMFLOAT3 VoxelTextureDimension = DirectX::XMFLOAT3(768.0f, 768.0f, 768.0f);

	private:
		VoxelBufferManager m_voxelBufferManager;

		ConstantBufferVoxelCommons m_cbVoxelCommons;
		DirectX::GraphicsResource m_cbVoxelCommonsResource;

		DX12Lib::TypedBuffer m_vertexBuffer;

		D3D12_VIEWPORT m_voxelScreenViewport;
		D3D12_RECT m_voxelScissorRect;

		std::vector<DX12Lib::GPUBuffer*> m_uavBuffers;

		UINT64 m_numberOfVoxels = 0;
	};
}