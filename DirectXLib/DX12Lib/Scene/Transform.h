#pragma once
#include <DirectXMath.h>
#include "DX12Lib/Commons/MathHelper.h"

namespace DX12Lib
{
	struct Transform
	{
		friend class SceneNode;

	private:
		enum class DirtyFlags
		{
			Position = 1 << 0,
			Rotation = 1 << 1,
			Scale = 1 << 2,

			All = Position | Rotation | Scale
		};

	public:
		Transform() :
		m_relativePos(0.0f, 0.0f, 0.0f), m_relativeRot(0.0f, 0.0f, 0.0f, 1.0f), m_relativeScale(1.0f, 1.0f, 1.0f),
		m_worldPos(0.0f, 0.0f, 0.0f), m_worldRot(0.0f, 0.0f, 0.0f, 1.0f), m_worldScale(1.0f, 1.0f, 1.0f)
		{
			m_relativeWorld = MathHelper::Identity4x4();
			m_world = MathHelper::Identity4x4();
			m_parent = nullptr;
		}

		~Transform() { m_parent = nullptr; }

		DirectX::XMVECTOR GetRelativePosition() const { return XMLoadFloat3(&m_relativePos); }
		DirectX::XMVECTOR GetRelativeRotation() const { return XMLoadFloat4(&m_relativeRot); }
		DirectX::XMVECTOR GetRelativeScale() const { return XMLoadFloat3(&m_relativeScale); }

		DirectX::XMFLOAT3 GetRelativePosition3f() const { return m_relativePos; }
		DirectX::XMFLOAT4 GetRelativeRotation4f() const { return m_relativeRot; }
		DirectX::XMFLOAT3 GetRelativeRotationEuler3f() const { return QuaternionToEuler(m_relativeRot); }
		DirectX::XMFLOAT3 GetRelativeScale3f() const { return m_relativeScale; }

		DirectX::XMVECTOR GetWorldPosition();
		DirectX::XMVECTOR GetWorldRotation();
		DirectX::XMVECTOR GetWorldScale();

		DirectX::XMFLOAT3 GetWorldPosition3f();
		DirectX::XMFLOAT4 GetWorldRotation4f();
		DirectX::XMFLOAT3 GetWorldRotationEuler3f();
		DirectX::XMFLOAT3 GetWorldScale3f();

		DirectX::XMMATRIX GetRelativeWorld();
		DirectX::XMMATRIX GetWorld();

		DirectX::XMFLOAT3 GetRight3f();
		DirectX::XMFLOAT3 GetUp3f();
		DirectX::XMFLOAT3 GetForward3f();

		DirectX::XMVECTOR GetRight();
		DirectX::XMVECTOR GetUp();
		DirectX::XMVECTOR GetForward();

		void SetRelativePosition(DirectX::FXMVECTOR pos);
		void SetRelativeRotation(DirectX::FXMVECTOR rot);
		void SetRelativeScale(DirectX::FXMVECTOR scale);

		void SetRelativePosition(DirectX::XMFLOAT3 pos);
		void SetRelativeRotation(DirectX::XMFLOAT4 rot);
		void SetRelativeScale(DirectX::XMFLOAT3 scale);

		void SetWorldPosition(DirectX::FXMVECTOR pos);
		void SetWorldRotation(DirectX::FXMVECTOR rot);
		void SetWorldScale(DirectX::FXMVECTOR scale);

		void SetWorldPosition(DirectX::XMFLOAT3 pos);
		void SetWorldRotation(DirectX::XMFLOAT4 rot);
		void SetWorldScale(DirectX::XMFLOAT3 scale);

		void Update();

		void SetDirty(DirtyFlags flag);
		void SetDirty(uint8_t flags = 5);

	private:
		DirectX::XMFLOAT3 QuaternionToEuler(DirectX::XMFLOAT4 quaternion) const;



	private:
		Transform* m_parent = nullptr;

		DirectX::XMFLOAT3 m_relativePos;
		DirectX::XMFLOAT4 m_relativeRot;
		DirectX::XMFLOAT3 m_relativeScale;
		DirectX::XMFLOAT4X4 m_relativeWorld;

		DirectX::XMFLOAT3 m_worldPos;
		DirectX::XMFLOAT4 m_worldRot;
		DirectX::XMFLOAT3 m_worldScale;
		DirectX::XMFLOAT4X4 m_world;

		DirectX::XMFLOAT3 m_right = { 1.0f, 0.0f, 0.0f };
		DirectX::XMFLOAT3 m_up = { 0.0f, 1.0f, 0.0f };
		DirectX::XMFLOAT3 m_forward = { 0.0f, 0.0f, 1.0f };

		uint8_t m_dirtyFlags = 0;
		uint8_t m_dirtForFrame = 0;


	};
}



