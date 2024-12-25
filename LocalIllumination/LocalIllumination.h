#pragma once
#define _WINSOCKAPI_
#include "DX12Lib/Commons/D3DApp.h"
#include "LIScene.h"
#include "DX12Lib/Commons/NetworkManager.h"

#include "DX12Lib/DXWrapper/Fence.h"
#include "DX12Lib/DXWrapper/GPUBuffer.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"

#include "DX12Lib/Commons/ShadowMap.h"

#include "Keyboard.h"

#include "LightTransportTechnique.h"
#include "SceneDepthTechnique.h"
#include "RadianceFromNetworkTechnique.h"
#include "GaussianFilterTechnique.h"

#define NUM_BASIC_BUFFERS 5

namespace VOX
{
	class TechniqueData;
	class LightTransportTechnique;
}

namespace LI
{
	struct NetworkRadianceBufferInfo
	{
		std::shared_ptr<DX12Lib::UploadBuffer> buffer;
		UINT nFaces = 0;
		UINT ShouldReset = 0;
	};

	class LocalIlluminationApp : public DX12Lib::D3DApp
	{
	public:
		virtual void Initialize(DX12Lib::GraphicsContext& commandContext) override;
		virtual void Update(DX12Lib::GraphicsContext& commandContext) override;
		virtual void Draw(DX12Lib::GraphicsContext& commandContext) override;
		virtual void OnClose(DX12Lib::GraphicsContext& commandContext) override;

	private:
		void OnPacketReceived(const DX12Lib::NetworkPacket* packet);
		DX12Lib::AABB GetSceneAABBExtents();
		void CopyDataToBasicBuffer(UINT bufferIdx);
	private:
		bool m_usePBRMaterials = true;
		DX12Lib::NetworkHost m_networkClient;
		DirectX::Keyboard::KeyboardStateTracker m_kbTracker;

		UINT m_buffersInitialized = 0;

		std::shared_ptr<VOX::BufferManager> m_voxelBufferManager;
		std::shared_ptr<VOX::BufferManager> m_prefixSumBufferManager;

		std::queue<std::pair<std::shared_ptr<DX12Lib::UploadBuffer>, UINT64>> m_ReadyToWriteBuffers;
		std::queue<NetworkRadianceBufferInfo> m_ReadyToCopyBuffer;

		DX12Lib::ShadowCamera m_depthCamera;

		std::unique_ptr<DX12Lib::Fence> m_bufferFence;
		std::unique_ptr<DX12Lib::Fence> m_graphicsFence;

		UINT m_writeRadIx = 0;

		const char* packetHeaders[NUM_BASIC_BUFFERS] = {"INDRNK", "INDIDX", "CMPIDX", "CMPHSH", "OCCVOX"};

		std::uint8_t m_lastCameraBitMask = 0;
		std::uint8_t m_lastLightBitMask = 0;

		std::mutex m_vectorMutex;
		std::mutex m_mainThreadMutex;
		std::condition_variable m_mainThreadCV;
		bool m_isMainThreadReady = false;

		std::shared_ptr<VOX::TechniqueData> m_data;
		std::shared_ptr<VOX::SceneDepthTechnique> m_sceneDepthTechnique;
		std::shared_ptr<LI::RadianceFromNetworkTechnique> m_radianceFromNetworkTechnique;
		std::shared_ptr<VOX::LightTransportTechnique> m_lightTransportTechnique;
		std::shared_ptr<VOX::GaussianFilterTechnique> m_gaussianFilterTechnique;


		enum class ReceiveState
		{
			INITIALIZATION,
			BASIC_BUFFERS,
			RADIANCE,
		} m_receiveState = ReceiveState::INITIALIZATION;

		std::unique_ptr<DX12Lib::Fence> m_rasterFence;
		std::unique_ptr<DX12Lib::Fence> m_rtgiFence;
		UINT64 m_rasterFenceValue = 0;

		bool LightDispatched = false;

		float lerpDeltaTime = 0.0f;
		float lerpMaxTime = 0.5f;

		bool m_radianceReady = false;

		float sendPacketDeltaTime = 0.0f;

		bool m_wasLightningChanged = false;

	public:
		LocalIlluminationApp(HINSTANCE hInstance, DX12Lib::Scene* scene = nullptr) : D3DApp(hInstance, scene) {};
		LocalIlluminationApp(const LocalIlluminationApp& rhs) = delete;
		LocalIlluminationApp& operator=(const LocalIlluminationApp& rhs) = delete;
		~LocalIlluminationApp();
	};
}