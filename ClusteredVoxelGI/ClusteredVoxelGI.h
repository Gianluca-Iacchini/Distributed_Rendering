#pragma once
#include <DX12Lib/Commons/D3DApp.h>
#include <memory>
#include <DirectXMath.h>
#include "DX12Lib/DXWrapper/RootSignature.h"
#include "DX12Lib/DXWrapper/PipelineState.h"
#include "Helpers/CVGIDataTypes.h"
#include "DX12Lib/DXWrapper/GPUBuffer.h"
#include "DX12Lib/DXWrapper/DescriptorHeap.h"
#include "GraphicsMemory.h"
#include "Techniques/Technique.h"

#include "Techniques/VoxelizeScene.h"
#include "Techniques/DisplayVoxelScene.h"
#include "Techniques/PrefixSumVoxels.h"
#include "Techniques/ClusterVoxels.h"
#include "Techniques/MergeClusters.h"
#include "Techniques/ComputeNeighboursTechnique.h"
#include "Techniques/ClusterVisibility.h"
#include "Techniques/BuildAABBsTechnique.h"
#include "Techniques/FacePenaltyTechnique.h"
#include "DX12Lib/Scene/LightComponent.h"

#include "Techniques/LightTransportTechnique.h"
#include "Techniques/GaussianFilterTechnique.h"
#include "Techniques/LerpRadianceTechnique.h"
#include "Techniques/SceneDepthTechnique.h"

#include "Techniques/LightVoxel.h"
#include "thread"

#include "DX12Lib/DXWrapper/Fence.h"

#include "DX12Lib/Commons/NetworkManager.h"

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
		virtual void OnClose(DX12Lib::GraphicsContext& commandContext) override;


	private:
		bool IsDirectXRaytracingSupported() const;
		void ComputeRTGI();

	public:
		//const DirectX::XMFLOAT3 VoxelTextureDimension = DirectX::XMFLOAT3(512.0f, 512.0f, 512.0f);
		const DirectX::XMFLOAT3 VoxelTextureDimension = DirectX::XMFLOAT3(256.0f, 256.0f, 256.0f);

	private:
		std::unique_ptr<VoxelizeScene> m_voxelizeScene;
		std::unique_ptr<DisplayVoxelScene> m_displayVoxelScene;
		std::unique_ptr<PrefixSumVoxels> m_prefixSumVoxels;
		std::unique_ptr<ClusterVoxels> m_clusterVoxels;
		std::unique_ptr<MergeClusters> m_mergeClusters;
		std::unique_ptr<ComputeNeighboursTechnique> m_computeNeighboursTechnique;
		std::unique_ptr<ClusterVisibility> m_clusterVisibility;
		std::unique_ptr<BuildAABBsTechnique> m_buildAABBsTechnique;
		std::unique_ptr<FacePenaltyTechnique> m_facePenaltyTechnique;
		std::unique_ptr<SceneDepthTechnique> m_sceneDepthTechnique;
		std::unique_ptr<LightVoxel> m_lightVoxel;
		std::unique_ptr<LightTransportTechnique> m_lightTransportTechnique;
		std::unique_ptr<GaussianFilterTechnique> m_gaussianFilterTechnique;
		std::unique_ptr<LerpRadianceTechnique> m_lerpRadianceTechnique;

		std::unique_ptr<DX12Lib::Fence> m_rtgiFence;
		std::unique_ptr<DX12Lib::Fence> m_rasterFence;
		std::unique_ptr<DX12Lib::Fence> m_blockFence;
		std::unique_ptr<DX12Lib::Fence> m_shadowFence;

		UINT32 m_rasterFenceValue = 0;
		UINT32 m_rtgiFenceValue = 0;

		UINT32 DirectLightFenceValue = 0;
		UINT32 IndirectLightFenceValue = 0;

		UINT32 IndirectBlockCount = 0;

		UINT32 m_lastBlockFenceVal = 0;
		UINT32 maxDispatchesPerFrame = 4;

		bool LightDispatched = false;
		bool BufferSwapped = true;
		bool m_isRunning = true;

		bool m_resetTime = false;
		bool m_resetCamera = false;

		bool changeLerp = false;

		DirectX::GraphicsResource m_cbVoxelCommonsResource;

		DX12Lib::TypedBuffer m_vertexBuffer;

		D3D12_VIEWPORT m_voxelScreenViewport;
		D3D12_RECT m_voxelScissorRect;

		std::vector<DX12Lib::GPUBuffer*> m_uavBuffers;

		UINT64 m_numberOfVoxels = 0;

		DX12Lib::ShadowCamera* m_shadowCamera = nullptr;

		std::shared_ptr<TechniqueData> m_data = nullptr;

		DX12Lib::NetworkHost m_networkServer;

		float RTGIUpdateDelta = 1.0f;
		float lerpDelta = 0.0f;

		float m_lastTotalTime = 0.0f;
	};
}