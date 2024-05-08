#pragma once

#include "MathHelper.h"

namespace DX12Lib {

#define MaxLights 16

	__declspec(align(16)) struct Light
	{
		DirectX::XMFLOAT3 Color = { 0.5f, 0.5f, 0.5f }; // All lights
		float FalloffStart = 1.0f; // Point, spot
		DirectX::XMFLOAT3 Direction = { 0.0f, -1.0f, 0.0f }; // Directional, spot
		float FalloffEnd = 10.0f; // Point, spot
		DirectX::XMFLOAT3 Position = { 0.0f, 0.0f, 0.0f }; // Point, spot
		float SpotPower = 64.0f; // Spot
	};

	__declspec(align(16)) struct CostantBufferCommons
	{
		DirectX::XMFLOAT4X4 view = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 invView = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 projection = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 invProjection = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 viewProjection = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 invViewProjection = MathHelper::Identity4x4();
		DirectX::XMFLOAT3 eyePosition = { 0.0f, 0.0f, 0.0f };
		float nearPlane = 0.0f;
		DirectX::XMFLOAT2 renderTargetSize = { 0.0f, 0.0f };
		DirectX::XMFLOAT2 invRenderTargetSize = { 0.0f, 0.0f };
		float farPlane = 1.0f;
		float totalTime = 0.0f;
		float deltaTime = 0.0f;

		Light lights[MaxLights];
	};

	__declspec(align(16)) struct ConstantBufferObject
	{
		DirectX::XMFLOAT4X4 world = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 invWorld = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 texTransform = MathHelper::Identity4x4();
		UINT materialIndex = 1;
	};
}



