#include "DX12Lib/Scene/Component.h"
#include "DirectXMath.h"

namespace DX12Lib
{

	class CameraController : public DX12Lib::Component
	{
	public:

	public:
		CameraController() : Component() {}

		void Init(DX12Lib::CommandContext& context) override;
		void Update(DX12Lib::CommandContext& context) override;
		DirectX::XMFLOAT3 GetVelocity();
	private:
		void Move(float speed, float deltaTime);

	private:
		float m_speed = 3.0f;
		DirectX::XMFLOAT3 m_lastPosition;
	};
}