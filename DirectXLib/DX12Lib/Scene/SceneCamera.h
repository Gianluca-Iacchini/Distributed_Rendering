#include "DX12Lib/Commons/Camera.h"
#include "DX12Lib/Scene/Component.h"
#include "DX12Lib/Commons/CommonConstants.h"

namespace DX12Lib
{
	class SceneCamera : public Camera, public Component
	{
	public:
		SceneCamera() : Camera(), Component() {};

		void Init(CommandContext& context) override;
		void Update(CommandContext& context) override;
		void Render(CommandContext& context) override;
		void OnResize(CommandContext& context) override;

		void UseCamera(CommandContext& context);

	private:
		DirectX::XMFLOAT3 m_lastPosition = { 0.0f, 0.0f, 0.0f };
		ConstantBufferCamera m_constantBufferCamera;
	};
}