#pragma once

#include "Texture.h"
#include "assimp/material.h"
#include "GraphicsMemory.h"
#include <DirectXMath.h>
#include "DX12Lib/DescriptorHeap.h"



namespace DX12Lib {

	enum class MaterialTextureType
	{
		DIFFUSE = 0,
		SPECULAR = 1,
		AMBIENT,
		EMISSIVE,
		SHININESS,
		NORMAL_MAP,
		BUMP_MAP,
		NUM_TEXTURE_TYPES
	};

	enum class PBRMaterialTextureType
	{
		ALBEDO = 0,
		EMISSIVE,
		NORMAL,
		METALLIC,
		ROUGHNESS,
		OCCLUSION,
		NUM_TEXTURE_TYPES
	};

	class MaterialManager;
	class MaterialBuilder;

	__declspec(align(16)) struct MaterialConstant
	{
	public:
		MaterialConstant(
			const DirectX::XMFLOAT4 diffuseColor = { 1.0f, 1.0f, 1.0f, 1.0f },
			const DirectX::XMFLOAT4 specularColor = { 1.0f, 1.0f, 1.0f, 1.0f },
			const DirectX::XMFLOAT4 ambientColor = { 0.0f, 0.0f, 0.0f, 1.0f },
			const DirectX::XMFLOAT4 emissiveColor = { 0.0f, 0.0f, 0.0f, 1.0f },
			const float opacity = 1.0f,
			const float shininess = 128.0f,
			const float indexOfRefraction = 1.0f,
			const float bumpIntensity = 1.0f)
			:
			DiffuseColor(diffuseColor),
			SpecularColor(specularColor),
			AmbientColor(ambientColor),
			EmissiveColor(emissiveColor),
			Opacity(opacity),
			Shininess(shininess),
			IndexOfRefraction(indexOfRefraction),
			BumpIntensity(bumpIntensity)
		{}

	public:

		DirectX::XMFLOAT4 DiffuseColor = { 1.0f, 1.0f, 1.0f, 1.0f };
		DirectX::XMFLOAT4 SpecularColor = { 1.0f, 1.0f, 1.0f, 1.0f };
		DirectX::XMFLOAT4 AmbientColor = { 0.0f, 0.0f, 0.0f, 1.0f };
		DirectX::XMFLOAT4 EmissiveColor = { 0.0f, 0.0f, 0.0f, 1.0f };

		float Opacity = 1.0f;
		float Shininess = 128.0f;
		float IndexOfRefraction = 1.0f;
		float BumpIntensity = 1.0f;
	};

	class Material
	{
		friend class MaterialManager;
		friend class MaterialBuilder;

	public:
		Material() = default;

		DirectX::XMFLOAT4 DiffuseColor = { 1.0f, 1.0f, 1.0f, 1.0f };
		DirectX::XMFLOAT4 SpecularColor = { 1.0f, 1.0f, 1.0f, 1.0f };
		DirectX::XMFLOAT4 AmbientColor = { 0.0f, 0.0f, 0.0f, 1.0f };
		DirectX::XMFLOAT4 EmissiveColor = { 0.0f, 0.0f, 0.0f, 1.0f };

		float Opacity = 1.0f;
		float Shininess = 128.0f;
		float IndexOfRefraction = 1.0f;
		float BumpIntensity = 1.0f;

	public:
		void UseMaterial(ID3D12GraphicsCommandList* cmdList);
		std::wstring& GetName() { return m_name; }

	private:
		MaterialConstant CreateMaterialConstant()
		{
			return MaterialConstant(
				DiffuseColor,
				SpecularColor,
				AmbientColor,
				EmissiveColor,
				Opacity,
				Shininess,
				IndexOfRefraction,
				BumpIntensity
			);
		}

		DirectX::GraphicsResource CreateMaterialBuffer();

	private:
		std::wstring m_name;
		SharedTexture m_textures[(UINT)MaterialTextureType::NUM_TEXTURE_TYPES];
		DescriptorHandle m_textureSRVHandles[(UINT)MaterialTextureType::NUM_TEXTURE_TYPES];
	};

	class PBRMaterial : public Material
	{
	public:
		PBRMaterial() = default;

	};

	using SharedMaterial = std::shared_ptr<Material>;

	class MaterialBuilder
	{
		friend class MaterialManager;

	public:

		void AddTexture(aiTextureType assimpTextureType, aiString& texturePath);
		void AddTexture(MaterialTextureType textureType, SharedTexture texture = nullptr);
		SharedTexture GetDefaultTextureForType(MaterialTextureType textureType);
		SharedMaterial BuildFromAssimpMaterial(aiMaterial* assimpMaterial, DescriptorHeap* textureHeap = nullptr);
		SharedMaterial Build(std::wstring& materialName, DescriptorHeap* textureHeap = nullptr);

		void SetDiffuseColor(DirectX::XMFLOAT4 diffuseColor) { m_material->DiffuseColor = diffuseColor; }
		void SetSpecularColor(DirectX::XMFLOAT4 specularColor) { m_material->SpecularColor = specularColor; }
		void SetAmbientColor(DirectX::XMFLOAT4 ambientColor) { m_material->AmbientColor = ambientColor; }
		void SetEmissiveColor(DirectX::XMFLOAT4 emissiveColor) { m_material->EmissiveColor = emissiveColor; }

		void SetOpacity(float opacity) { m_material->Opacity = opacity; }
		void SetShininess(float shininess) { m_material->Shininess = shininess; }
		void SetIndexOfRefraction(float indexOfRefraction) { m_material->IndexOfRefraction = indexOfRefraction; }
		void SetBumpIntensity(float bumpIntensity) { m_material->BumpIntensity = bumpIntensity; }

		MaterialTextureType AssimpToTextureType(aiTextureType assimpTextureType)
		{
			switch (assimpTextureType)
			{
			case aiTextureType_DIFFUSE:
				return MaterialTextureType::DIFFUSE;
			case aiTextureType_SPECULAR:
				return MaterialTextureType::SPECULAR;
			case aiTextureType_AMBIENT:
				return MaterialTextureType::AMBIENT;
			case aiTextureType_EMISSIVE:
				return MaterialTextureType::EMISSIVE;
			case aiTextureType_SHININESS:
				return MaterialTextureType::SHININESS;
			case aiTextureType_NORMALS:
				return MaterialTextureType::NORMAL_MAP;
			case aiTextureType_HEIGHT:
				return MaterialTextureType::BUMP_MAP;
			default:
				return MaterialTextureType::DIFFUSE;
			}
		}



	private:
		MaterialBuilder(MaterialManager* manager) : m_materialManager(manager)
		{
			m_material = std::make_shared<Material>();
		}

		void LoadAssimpTextures(aiMaterial* assimpMaterial, DescriptorHeap* textureHeap);
		void LoadAssimpConstants(aiMaterial* assimpMaterial);

	private:
		SharedMaterial m_material;
		MaterialManager* m_materialManager;
	};


	class MaterialManager
	{
		friend class MaterialBuilder;

	public:
		MaterialManager() = default;
		//SharedMaterial LoadMaterial(const )
		MaterialBuilder CreateMaterialBuilder() { return MaterialBuilder(this); }

		DirectX::GraphicsResource CreateMaterialBuffer(Material* const material);

		SharedMaterial GetMaterial(std::wstring& materialName);
		void AddMaterial(SharedMaterial material);
		void RemoveMaterial(SharedMaterial material);

	private:
		void LoadDefaultMaterials(TextureManager& textureManager);

		std::mutex m_materialCacheMutex;
		std::unordered_map<std::wstring, SharedMaterial> m_materialCache;
	};
}
