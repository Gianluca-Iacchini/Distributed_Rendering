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

#define NUM_BASIC_BUFFERS 5

namespace VOX
{
	class TechniqueData;
	class LightTransportTechnique;
}

namespace LI
{

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
	private:
		bool m_usePBRMaterials = true;
		DX12Lib::NetworkHost m_networkClient;
		DirectX::Keyboard::KeyboardStateTracker m_kbTracker;

		ConstantBufferVoxelCommons m_cbVoxelCommons;

		UINT m_buffersInitialized = 0;
		UINT m_voxelCount = 0;
		UINT m_faceCount = 0;
		UINT m_clusterCount = 0;

		DX12Lib::UploadBuffer m_uploadBuffer;
		DX12Lib::StructuredBuffer m_basicBuffers[5];
		DX12Lib::StructuredBuffer m_radianceRingBuffers[3];

		DX12Lib::ShadowCamera m_depthCamera;

		std::unique_ptr<DX12Lib::Fence> m_bufferFence;
		std::unique_ptr<DX12Lib::Fence> m_graphicsFence;

		UINT m_writeRadIx = 0;

		const char* packetHeaders[NUM_BASIC_BUFFERS] = { "OCCVOX", "INDRNK", "INDIDX", "CMPIDX", "CMPHSH" };

		std::uint8_t m_lastInputBitMask = 0;

		std::queue<std::pair<UINT, UINT64>> m_fenceForBufferIdx;

		std::mutex m_vectorMutex;

		std::shared_ptr<VOX::TechniqueData> m_data;
		

		enum class ReceiveState
		{
			INITIALIZATION,
			BASIC_BUFFERS,
			RADIANCE,
		} m_receiveState = ReceiveState::INITIALIZATION;


	public:
		LocalIlluminationApp(HINSTANCE hInstance, DX12Lib::Scene* scene = nullptr) : D3DApp(hInstance, scene) {};
		LocalIlluminationApp(const LocalIlluminationApp& rhs) = delete;
		LocalIlluminationApp& operator=(const LocalIlluminationApp& rhs) = delete;
		~LocalIlluminationApp();
	};
}