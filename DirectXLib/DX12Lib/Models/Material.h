#pragma once

#include "DX12Lib/DXWrapper/Texture.h"
#include "assimp/material.h"
#include "GraphicsMemory.h"
#include <DirectXMath.h>
#include "DX12Lib/DXWrapper/DescriptorHeap.h"
#include "DX12Lib/Commons/CommonConstants.h"

namespace DX12Lib {

#define NUM_COMMON_TEXTURES ((UINT)MaterialTextureType::NORMAL_MAP + 1)
#define NUM_PHONG_TEXTURES ((UINT)MaterialTextureType::BUMP_MAP + 1)
#define NUM_PBR_TEXTURES ((UINT)MaterialTextureType::NUM_TEXTURE_TYPES - NUM_PHONG_TEXTURES + NUM_COMMON_TEXTURES)

#define COMMON_TEXTURE_OFFSET 0
#define PHONG_TEXTURE_OFFSET NUM_COMMON_TEXTURES
#define PBR_TEXTURE_OFFSET NUM_PHONG_TEXTURES

	enum class MaterialTextureType
	{
		EMISSIVE = 0,
		NORMAL_MAP = 1,
		DIFFUSE,
		SPECULAR,
		AMBIENT,
		SHININESS,
		BUMP_MAP,
		BASECOLOR,
		METALROUGHNESS,
		OCCLUSION,
		NUM_TEXTURE_TYPES
	};

	class MaterialManager;
	class MaterialBuilder;

	enum class MaterialShadingModel
	{
		PHONG,
		PBR,
		UNLIT,
		UNKNOWN
	};

	class Material
	{
		friend class MaterialManager;
		friend class MaterialBuilder;

	public:
		Material();

		virtual ~Material()
		{
			if (m_textures != nullptr)
				delete[] m_textures;
		}

		void UseMaterial(ID3D12GraphicsCommandList* cmdList);
		std::wstring& GetName() { return m_name; }

		virtual void SetTexture(MaterialTextureType type, SharedTexture texture) {}

		virtual MaterialShadingModel ShadingModel() { return MaterialShadingModel::UNKNOWN; }

		std::wstring GetDefaultPSO() { return m_defaultPSO; }

		virtual ConstantBufferMaterial BuildMaterialConstantBuffer() { return ConstantBufferMaterial(); }
	public:
		DirectX::XMFLOAT4 DiffuseColor = { 1.0f, 1.0f, 1.0f, 1.0f };
		DirectX::XMFLOAT4 EmissiveColor = { 0.0f, 0.0f, 0.0f, 1.0f };


		float NormalScale = 1.0f;

	protected:
		std::wstring m_name;
		SharedTexture* m_textures = nullptr;
		DescriptorHandle m_firstTextureHandle;

		std::wstring m_defaultPSO;
	};

	class PhongMaterial : public Material
	{
	public:
		PhongMaterial();


		virtual void SetTexture(MaterialTextureType type, SharedTexture texture) override;

		virtual MaterialShadingModel ShadingModel() override { return MaterialShadingModel::PHONG; }

		virtual ConstantBufferMaterial BuildMaterialConstantBuffer() override;
	public:
		DirectX::XMFLOAT4 SpecularColor = { 1.0f, 1.0f, 1.0f, 1.0f };
		DirectX::XMFLOAT4 AmbientColor = { 0.0f, 0.0f, 0.0f, 1.0f };

		float Opacity = 1.0f;
		float Shininess = 128.0f;
		float IndexOfRefraction = 1.0f;

	protected:
		
	};

	class PBRMaterial : public Material
	{
	public:
		PBRMaterial();

		virtual void SetTexture(MaterialTextureType type, SharedTexture texture) override;

		virtual MaterialShadingModel ShadingModel() override { return MaterialShadingModel::PBR; }

		virtual ConstantBufferMaterial BuildMaterialConstantBuffer() override;
	public:
		float Metallic = 0.2f;
		float Roughness = 0.4f;

	};


}
