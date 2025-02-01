#pragma once
#include <DX12Lib/Commons/D3DApp.h>
#include "DX12Lib/DXWrapper/QueryHeap.h"
#include <memory>
#include <DirectXMath.h>
#include "DX12Lib/DXWrapper/RootSignature.h"
#include "DX12Lib/DXWrapper/PipelineState.h"
#include "../VoxelUtils/CVGIDataTypes.h"
#include "DX12Lib/DXWrapper/GPUBuffer.h"
#include "DX12Lib/DXWrapper/DescriptorHeap.h"
#include "GraphicsMemory.h"
#include "Techniques/Technique.h"

#include "VoxelScene.h"

#include "Techniques/VoxelizeScene.h"
#include "Techniques/DisplayVoxelScene.h"
#include "Techniques/PrefixSumVoxels.h"
#include "Techniques/ClusterVoxels.h"
#include "Techniques/ComputeNeighboursTechnique.h"
#include "Techniques/ClusterVisibility.h"
#include "Techniques/BuildAABBsTechnique.h"
#include "DX12Lib/Scene/LightComponent.h"

#include "Techniques/LightTransportTechnique.h"
#include "Techniques/GaussianFilterTechnique.h"
#include "Techniques/SceneDepthTechnique.h"

#include "Techniques/LightVoxel.h"
#include "thread"

#include "DX12Lib/DXWrapper/Fence.h"

#include "NetworkManager.h"


namespace CVGI
{
	class VoxelCamera;

	enum class IMGUIWindowStatus
	{
		VOXEL_SELECTION_SCREEN = 0,
		LOADING_SCREEN = 1,
		INITIALIZING = 2,
		VOXEL_DEBUG_SCREEN = 3,
	};

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
		bool ShowIMGUIWindow(DX12Lib::GraphicsContext& context);
		bool ShowIMGUIVoxelOptionWindow(float appX, float appY);
		void ShowIMGUILoadingWindow(float appX, float appY);
		void ShowIMGUIVoxelDebugWindow(float appX, float appY);
		void InitializeVoxelData(DX12Lib::GraphicsContext& commandContext);
		void OnPacketReceived(const Commons::NetworkPacket* packet);
		void OnClientConnected(const ENetPeer* peer);
		void OnClientDisconnected(const ENetPeer* peer);
		bool IsDirectXRaytracingSupported() const;

	public:
		//const DirectX::XMFLOAT3 VoxelTextureDimension = DirectX::XMFLOAT3(512.0f, 512.0f, 512.0f);
		DirectX::XMUINT3 VoxelTextureDimension = DirectX::XMUINT3(256.0f, 256.0f, 256.0f);

	private:
		std::unique_ptr<VoxelizeScene> m_voxelizeScene;
		std::unique_ptr<DisplayVoxelScene> m_displayVoxelScene;
		std::unique_ptr<PrefixSumVoxels> m_prefixSumVoxels;
		std::unique_ptr<ClusterVoxels> m_clusterVoxels;
		std::unique_ptr<ComputeNeighboursTechnique> m_computeNeighboursTechnique;
		std::unique_ptr<ClusterVisibility> m_clusterVisibility;
		std::unique_ptr<BuildAABBsTechnique> m_buildAABBsTechnique;
		std::unique_ptr<VOX::SceneDepthTechnique> m_sceneDepthTechnique;
		std::unique_ptr<LightVoxel> m_lightVoxel;
		std::unique_ptr<VOX::LightTransportTechnique> m_lightTransportTechnique;
		std::unique_ptr<VOX::GaussianFilterTechnique> m_gaussianFilterTechnique;

		std::unique_ptr<DX12Lib::Fence> m_rtgiFence;
		std::unique_ptr<DX12Lib::Fence> m_rasterFence;

		UINT32 m_rasterFenceValue = 0;
		UINT32 m_rtgiFenceValue = 0;
		UINT32 m_lastCompletedFenceValue = 0;

		CVGI::VoxelScene* m_voxelScene = nullptr;

		DX12Lib::QueryHandle m_timingQueryHandle;
		DX12Lib::ReadBackBuffer m_timingReadBackBuffer;

		UINT64 m_rtgiMemoryUsage = 0;

		float m_voxelBuildTime = 0.0f;
		float m_prefixSumTime = 0.0f;
		float m_clusterizeTime = 0.0f;
		float m_computeNeighboursTime = 0.0f;
		float m_buildingAccelerationStructuresTime = 0.0f;
		float m_clusterVisibilityTime = 0.0f;
		float m_buildAABBsTime = 0.0f;
		float m_initialRadianceTime = 0.0f;


		float m_litVoxelTime = 0.0f;
		float m_visibleVoxelTime = 0.0f;
		float m_computeRadianceTime = 0.0f;
		float m_firstGaussianFilterTime = 0.0f;
		float m_secondGaussianFilterTime = 0.0f;

		float m_accTotalTime = 0.0f;
		UINT64 m_lightDispatchCount = 0;

		bool LightDispatched = false;
		bool BufferSwapped = true;
		bool m_isRunning = true;

		bool m_resetTime = false;

		bool m_cameraMovedSinceLastUpdate = true;
		bool m_lightChangedSinceLastUpdate = false;

		std::atomic<bool> m_clientLightUpdate = false;
		std::atomic<bool> m_clientCameraUpdate = false;

		bool m_isRadianceReady = false;

		UINT32 m_wasRadianceReset = 0;

		DirectX::GraphicsResource m_cbVoxelCommonsResource;

		DX12Lib::TypedBuffer m_vertexBuffer;

		D3D12_VIEWPORT m_voxelScreenViewport;
		D3D12_RECT m_voxelScissorRect;

		UINT64 m_numberOfVoxels = 0;

		DX12Lib::ShadowCamera* m_shadowCamera = nullptr;

		std::shared_ptr<VOX::TechniqueData> m_data = nullptr;

		Commons::NetworkHost m_networkServer;

		char m_serverAddress[16] = { "87.14.75.14" };

		DirectX::XMFLOAT3 m_lastCameraPosition = DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f);

		bool m_isClientReadyForRadiance = false;
		bool m_firstRadianceSent = false;

		bool m_renderRasterScene = true;

		float m_RTGIUpdateDelta = 0.0f;
		float m_RTGIMaxTime = 0.15f;

		float m_lerpDeltaTime = 0.0f;
		float m_lerpMaxTime = 0.2f;

		DX12Lib::ComputeContext* m_computeContext = nullptr;
	};
}