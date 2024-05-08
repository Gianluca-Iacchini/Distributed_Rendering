#pragma once

#include <DirectXMath.h>

namespace DX12Lib {

	class Camera
	{
	public:
		Camera();
		~Camera();

		DirectX::XMVECTOR GetPosition() const;
		DirectX::XMFLOAT3 GetPosition3f() const;
		void SetPosition(float x, float y, float z);
		void SetPosition(const DirectX::XMFLOAT3& v);

		DirectX::XMVECTOR GetRight() const;
		DirectX::XMFLOAT3 GetRight3f() const;
		DirectX::XMVECTOR GetUp() const;
		DirectX::XMFLOAT3 GetUp3f() const;
		DirectX::XMVECTOR GetLook() const;
		DirectX::XMFLOAT3 GetLook3f() const;

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

		void LookAt(DirectX::FXMVECTOR pos, DirectX::FXMVECTOR target, DirectX::FXMVECTOR worldUp);
		void LookAt(const DirectX::XMFLOAT3& pos, const DirectX::XMFLOAT3& target, const DirectX::XMFLOAT3& worldUp);

		DirectX::XMMATRIX GetView() const;
		DirectX::XMMATRIX GetProjection() const;

		DirectX::XMFLOAT4X4 GetView4x4f() const;
		DirectX::XMFLOAT4X4 GetProjection4x4f() const;

		void Strafe(float d);
		void Walk(float d);
		void Lift(float d);

		void Pitch(float angle);
		void Yaw(float angle);

		void UpdateViewMatrix();
	private:
		DirectX::XMFLOAT3 m_position = { 0.0f, 0.0f, 0.0f };
		DirectX::XMFLOAT3 m_right = { 1.0f, 0.0f, 0.0f };
		DirectX::XMFLOAT3 m_up = { 0.0f, 1.0f, 0.0f };
		DirectX::XMFLOAT3 m_look = { 0.0f, 0.0f, 1.0f };

		float m_nearZ = 0.0f;
		float m_farZ = 0.0f;
		float m_aspect = 0.0f;
		float m_fovY = 0.0f;
		float m_nearWindowHeight = 0.0f;
		float m_farWindowHeight = 0.0f;

		bool m_viewDirty = true;

		DirectX::XMFLOAT4X4 m_view = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 m_proj = MathHelper::Identity4x4();
	};
}