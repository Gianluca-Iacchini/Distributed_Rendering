#pragma once

#include "DX12Lib/DXWrapper/Texture.h"
#include <DirectXMath.h>
#include "DX12Lib/DXWrapper/DescriptorHeap.h"
#include "DX12Lib/Commons/CommonConstants.h"

namespace DX12Lib {

#define NUM_COMMON_TEXTURES ((UINT)MaterialTextureType::DIFFUSE)
#define NUM_PHONG_TEXTURES ((UINT)MaterialTextureType::BASECOLOR)
#define NUM_PBR_TEXTURES ((UINT)MaterialTextureType::NUM_TEXTURE_TYPES - NUM_PHONG_TEXTURES + NUM_COMMON_TEXTURES)

#define COMMON_TEXTURE_OFFSET 0
#define PHONG_TEXTURE_OFFSET NUM_COMMON_TEXTURES
#define PBR_TEXTURE_OFFSET NUM_PHONG_TEXTURES

#define IS_COMMON(tex) ((UINT)tex < NUM_COMMON_TEXTURES)
#define IS_PHONG(tex) ((UINT)tex < NUM_PHONG_TEXTURES)
#define IS_PBR(tex) (IS_COMMON(tex) || ((UINT)tex >= PBR_TEXTURE_OFFSET))

	enum class MaterialTextureType
	{
		EMISSIVE = 0,
		NORMAL_MAP = 1,
		DIFFUSE,
		SPECULAR,
		AMBIENT,
		SHININESS,
		OPACITY,
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
		void SetTexture(UINT index, SharedTexture texture);
		Texture* GetTexture(MaterialTextureType type);
		virtual Texture* GetTexture(UINT index);

		virtual MaterialShadingModel ShadingModel() { return MaterialShadingModel::UNKNOWN; }

		std::wstring GetDefaultPSO() { return m_defaultPSO; }

		virtual ConstantBufferMaterial BuildMaterialConstantBuffer() { return ConstantBufferMaterial(); }

		virtual void SetTransparent(bool isTransparent);

		bool IsTransparent() { return m_isTransparent; }

		virtual UINT GetTextureCount() { return 0; }

		virtual void LoadDefaultTextures() {}

	protected:
		SharedTexture GetDefaultTextureForType(MaterialTextureType textureType);

	public:
		DirectX::XMFLOAT4 DiffuseColor = { 1.0f, 1.0f, 1.0f, 1.0f };
		DirectX::XMFLOAT4 EmissiveColor = { 0.0f, 0.0f, 0.0f, 1.0f };


		float NormalScale = 1.0f;



	protected:
		std::wstring m_name;
		SharedTexture* m_textures = nullptr;
		DescriptorHandle m_firstTextureHandle;
		bool m_isTransparent = false;
		std::wstring m_defaultPSO;
	};

	class PhongMaterial : public Material
	{
	public:
		PhongMaterial();


		virtual void SetTexture(MaterialTextureType type, SharedTexture texture) override;
		virtual Texture* GetTexture(UINT index) override;

		virtual MaterialShadingModel ShadingModel() override { return MaterialShadingModel::PHONG; }

		virtual ConstantBufferMaterial BuildMaterialConstantBuffer() override;

		virtual void SetTransparent(bool isTransparent) override;
		virtual UINT GetTextureCount() override { return NUM_PHONG_TEXTURES; }
		virtual void LoadDefaultTextures() override;

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
		Texture* GetTexture(UINT index) override;

		virtual MaterialShadingModel ShadingModel() override { return MaterialShadingModel::PBR; }

		virtual ConstantBufferMaterial BuildMaterialConstantBuffer() override;

		virtual void SetTransparent(bool isTransparent) override;

		virtual UINT GetTextureCount() override { return NUM_PBR_TEXTURES; }

		virtual void LoadDefaultTextures() override;

	public:
		float Metallic = 0.2f;
		float Roughness = 0.4f;

	};


}
