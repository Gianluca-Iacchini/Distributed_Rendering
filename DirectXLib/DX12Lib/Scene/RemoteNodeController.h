#pragma once

#include "DX12Lib/Scene/Component.h"

namespace DX12Lib
{
	class RemoteNodeController : public Component
	{
	public:
		RemoteNodeController();
		virtual ~RemoteNodeController() {}

		void SetRemoteControl(bool isRemoteControlled) { m_isRemoteControlled = isRemoteControlled; }
		bool IsRemoteControlled() { return m_isRemoteControlled; }

		void FeedRemoteData(DirectX::XMFLOAT3 velocity, DirectX::XMFLOAT3 absPos, DirectX::XMFLOAT4 absRot, UINT64 timestamp);
		void FeedRemoteData(DirectX::XMFLOAT2 mouseDeltaXY, std::uint8_t inputBitmap, UINT64 timestamp, float clientDeltaTime);

		virtual void Update(DX12Lib::CommandContext& context) override;

	private:
		bool m_isRemoteControlled = false;
		DirectX::XMFLOAT3 m_lastVelocity;
	};
}