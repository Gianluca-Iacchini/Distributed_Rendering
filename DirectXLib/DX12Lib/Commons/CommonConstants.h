#pragma once

#include "MathHelper.h"

namespace DX12Lib {

#define MaxLights 16

	__declspec(align(16)) struct ConstantBufferLight
	{
		DirectX::XMFLOAT3 Color = { 0.5f, 0.5f, 0.5f }; // All lights
		float FalloffStart = 1.0f; // Point, spot
		DirectX::XMFLOAT3 Direction = { 0.0f, -1.0f, 0.0f }; // Directional, spot
		float FalloffEnd = 10.0f; // Point, spot
		DirectX::XMFLOAT3 Position = { 0.0f, 0.0f, 0.0f }; // Point, spot
		float SpotPower = 64.0f; // Spot
		DirectX::XMFLOAT4X4 shadowTransform = MathHelper::Identity4x4();
		int CastsShadows = 0;
		float _pad1 = 0.0f;
		float _pad2 = 0.0f;
		float _pad3 = 0.0f;
	};

	__declspec(align(16)) struct CostantBufferCommons
	{

		DirectX::XMFLOAT2 renderTargetSize = { 0.0f, 0.0f };
		DirectX::XMFLOAT2 invRenderTargetSize = { 0.0f, 0.0f };

		float totalTime = 0.0f;
		float deltaTime = 0.0f;
		int numLights = 0;
		UINT32 useRTGI = 0;
	};

	__declspec(align(16)) struct ConstantBufferVoxelTransform
	{
		DirectX::XMFLOAT4X4 voxelToWorld = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 worldToVoxel = MathHelper::Identity4x4();
	};

	__declspec(align(16)) struct ConstantBufferCamera
	{
		DirectX::XMFLOAT4X4 view = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 invView = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 projection = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 invProjection = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 viewProjection = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 invViewProjection = MathHelper::Identity4x4();
		// Use by shadow cameras only.
		DirectX::XMFLOAT3 eyePosition = { 0.0f, 0.0f, 0.0f };
		float nearPlane = 0.0f;

		float farPlane = 1.0f;
		float _pad0 = 0.0f;
		float _pad1 = 0.0f;
		float _pad2 = 0.0f;
	};

	__declspec(align(16)) struct ConstantBufferObject
	{
		DirectX::XMFLOAT4X4 world = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 invWorld = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 texTransform = MathHelper::Identity4x4();

		UINT materialIndex = 1;
		float _pad0 = 0.0f;
		float _pad1 = 0.0f;
		float _pad2 = 0.0f;
	};

	__declspec(align(16)) struct ConstantBufferMaterial
	{
		ConstantBufferMaterial(
			DirectX::XMFLOAT4 float4_0 = { 0.0f, 0.0f, 0.0f, 0.0f },
			DirectX::XMFLOAT4 float4_1 = { 0.0f, 0.0f, 0.0f, 0.0f },
			DirectX::XMFLOAT4 float4_2 = { 0.0f, 0.0f, 0.0f, 0.0f },
			DirectX::XMFLOAT4 float4_3 = { 0.0f, 0.0f, 0.0f, 0.0f },
			float float_0 = 0.0f,
			float float_1 = 0.0f,
			float float_2 = 0.0f,
			float float_3 = 0.0f
		)
			:
			Float4_0(float4_0),
			Float4_1(float4_1),
			Float4_2(float4_2),
			Float4_3(float4_3),
			Float_0(float_0),
			Float_1(float_1),
			Float_2(float_2),
			Float_3(float_3)
		{
		}
	public:
		DirectX::XMFLOAT4 Float4_0 = { 0.0f, 0.0f, 0.0f, 0.0f };
		DirectX::XMFLOAT4 Float4_1 = { 0.0f, 0.0f, 0.0f, 0.0f };
		DirectX::XMFLOAT4 Float4_2 = { 0.0f, 0.0f, 0.0f, 0.0f };
		DirectX::XMFLOAT4 Float4_3 = { 0.0f, 0.0f, 0.0f, 0.0f };

		float Float_0 = 0.0f;
		float Float_1 = 0.0f;
		float Float_2 = 0.0f;
		float Float_3 = 0.0f;
	};
}



