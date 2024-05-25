#pragma once

#include <DirectXMath.h>

namespace DX12Lib {

	class Camera
	{
	public:
		Camera();
		~Camera();

		float GetNearZ() const;
		float GetFarZ() const;
		float GetAspect() const;
		float GetFovY() const;
		float GetFovX() const;

		float GetNearWindowWidth() const;
		float GetNearWindowHeight() const;
		float GetFarWindowWidth() const;
		float GetFarWindowHeight() const;

		void SetLens(float fovY, float aspect, float zn, float zf);
		void SetOrthogonalBounds(DirectX::XMFLOAT3 center, DirectX::XMFLOAT3 halfExtents);
		void SetOrthogonalBounds(float width, float height, float nearZ, float farZ);

		DirectX::XMMATRIX GetView() const;
		DirectX::XMMATRIX GetProjection() const;



		DirectX::XMFLOAT4X4 GetView4x4f() const;
		DirectX::XMFLOAT4X4 GetProjection4x4f() const;

		void UpdateViewMatrix(DirectX::FXMVECTOR pos, DirectX::FXMVECTOR up, DirectX::FXMVECTOR forward,
			DirectX::GXMVECTOR right);

	protected:

		float m_nearZ = 0.0f;
		float m_farZ = 0.0f;
		float m_aspect = 0.0f;
		float m_fovY = 0.0f;
		float m_nearWindowHeight = 0.0f;
		float m_farWindowHeight = 0.0f;

		DirectX::XMFLOAT4X4 m_view = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 m_proj = MathHelper::Identity4x4();

	};
}