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


	__declspec(align(16)) struct ConstantBufferPhongMaterial
	{
	public:
		ConstantBufferPhongMaterial(
			const DirectX::XMFLOAT4 diffuseColor = { 1.0f, 1.0f, 1.0f, 1.0f },
			const DirectX::XMFLOAT4 specularColor = { 1.0f, 1.0f, 1.0f, 1.0f },
			const DirectX::XMFLOAT4 ambientColor = { 0.0f, 0.0f, 0.0f, 1.0f },
			const DirectX::XMFLOAT4 emissiveColor = { 0.0f, 0.0f, 0.0f, 1.0f },
			const float opacity = 1.0f,
			const float shininess = 128.0f,
			const float indexOfRefraction = 1.0f,
			const float normalScale = 1.0f)
			:
			DiffuseColor(diffuseColor),
			EmissiveColor(emissiveColor),
			SpecularColor(specularColor),
			AmbientColor(ambientColor),
			NormalScale(normalScale),
			Opacity(opacity),
			Shininess(shininess),
			IndexOfRefraction(indexOfRefraction)
		{
		}

	public:
		DirectX::XMFLOAT4 DiffuseColor = { 1.0f, 1.0f, 1.0f, 1.0f };
		DirectX::XMFLOAT4 EmissiveColor = { 0.0f, 0.0f, 0.0f, 1.0f };
		DirectX::XMFLOAT4 SpecularColor = { 1.0f, 1.0f, 1.0f, 1.0f };
		DirectX::XMFLOAT4 AmbientColor = { 0.0f, 0.0f, 0.0f, 1.0f };

		
		float NormalScale = 1.0f;
		float Opacity = 1.0f;
		float Shininess = 128.0f;
		float IndexOfRefraction = 1.0f;
	};

	__declspec(align(16)) struct ConstantBufferPBRMaterial
	{
	public:
		ConstantBufferPBRMaterial(
			const DirectX::XMFLOAT4 baseColor = { 1.0f, 1.0f, 1.0f, 1.0f },
			const DirectX::XMFLOAT4 emissiveColor = { 0.0f, 0.0f, 0.0f, 1.0f },
			const float roughness = 0.4f,
			const float metallic = 0.2f,
			const float normalScale = 1.0f
		)
			:
			BaseColor(baseColor),
			EmissiveColor(emissiveColor),
			Roughness(roughness),
			Metallic(metallic),
			NormalScale(normalScale)
		{
		}

	public:
		DirectX::XMFLOAT4 BaseColor = { 1.0f, 1.0f, 1.0f, 1.0f };
		DirectX::XMFLOAT4 EmissiveColor = { 0.0f, 0.0f, 0.0f, 1.0f };

		float Roughness = 0.4f;
		float Metallic = 0.2f;
		float NormalScale = 1.0f;
	};
}



