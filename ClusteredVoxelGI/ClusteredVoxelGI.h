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
#include "VoxelizeScene.h"
#include "PrefixSumVoxels.h"
#include "ClusterVoxels.h"
#include "MergeClusters.h"
#include "ClusterVisibility.h"

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


	private:

	public:
		const DirectX::XMFLOAT3 VoxelTextureDimension = DirectX::XMFLOAT3(512.0f, 512.0f, 512.0f);

	private:
		std::unique_ptr<VoxelizeScene> m_voxelizeScene;
		std::unique_ptr<PrefixSumVoxels> m_prefixSumVoxels;
		std::unique_ptr<ClusterVoxels> m_clusterVoxels;
		std::unique_ptr<MergeClusters> m_mergeClusters;
		std::unique_ptr<ClusterVisibility> m_clusterVisibility;


		ConstantBufferVoxelCommons m_cbVoxelCommons;
		DirectX::GraphicsResource m_cbVoxelCommonsResource;

		DX12Lib::TypedBuffer m_vertexBuffer;

		D3D12_VIEWPORT m_voxelScreenViewport;
		D3D12_RECT m_voxelScissorRect;

		std::vector<DX12Lib::GPUBuffer*> m_uavBuffers;

		UINT64 m_numberOfVoxels = 0;
	};
}