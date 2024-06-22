#pragma once

#include "DX12Lib/Commons/Camera.h"
#include "DX12Lib/Scene/Component.h"
#include "DX12Lib/Commons/CommonConstants.h"

namespace DX12Lib
{
	class SceneCamera : public Camera, public Component
	{
	public:
		SceneCamera() : Camera(), Component() {};
		virtual ~SceneCamera() = default;

		virtual void Init(CommandContext& context) override;
		virtual void Update(CommandContext& context) override;
		virtual void Render(CommandContext& context) override;
		virtual void OnResize(CommandContext& context, int newWidth, int newHeight) override;

		virtual void UseCamera(CommandContext& context);

		virtual void SetOrthogonal(DirectX::XMFLOAT4 bounds);
		virtual void SetPerspective(float fov, float aspectRatio, float nearZ, float farZ);
		virtual bool IsOrtohgraphic() { return m_isOrthographic; }
		virtual bool IsPerspective() { return !m_isOrthographic; }

	public:
		bool IsEnabled = true;


	private:
		DirectX::XMFLOAT3 m_lastPosition = { 0.0f, 0.0f, 0.0f };
		ConstantBufferCamera m_constantBufferCamera;

		int m_cameraForward = 0;
		int m_cameraStrafe = 0;
		int m_cameraLift = 0;

		float m_oldMouseX = 0.0f;
		float m_oldMouseY = 0.0f;

		UINT m_inputCounter = 0;

		bool m_isOrthographic = false;

		DirectX::XMFLOAT4 m_orthogonalBounds = { 1000.0f, 1000.0f, 0.1f, 1000.0f };
	};
}