#pragma once

#include "Material.h"
#include "assimp/material.h"

namespace DX12Lib
{
	using SharedMaterial = std::shared_ptr<Material>;

	class MaterialBuilder
	{
		friend class MaterialManager;

	public:

		void AddTexture(aiTextureType assimpTextureType, aiString& texturePath);
		void AddTexture(MaterialTextureType textureType, SharedTexture texture = nullptr);

		SharedMaterial BuildFromAssimpMaterial(aiMaterial* assimpMaterial);
		SharedMaterial Build(std::wstring materialName);

		void SetDiffuseColor(DirectX::XMFLOAT4 diffuseColor) { m_material->DiffuseColor = diffuseColor; }
		void SetSpecularColor(DirectX::XMFLOAT4 specularColor);
		void SetAmbientColor(DirectX::XMFLOAT4 ambientColor);
		void SetEmissiveColor(DirectX::XMFLOAT4 emissiveColor) { m_material->EmissiveColor = emissiveColor; }

		void SetOpacity(float opacity);
		void SetShininess(float shininess);
		void SetIndexOfRefraction(float indexOfRefraction);
		void SetNormalScale(float normalScale) { m_material->NormalScale = normalScale; }
		void SetRoughness(float roughness);
		void SetMetallic(float metallic);

		MaterialTextureType AssimpToTextureType(aiTextureType assimpTextureType);

		bool IsPBR() { return m_isPBR; }

	private:
		MaterialBuilder(MaterialManager* manager)
			: m_materialManager(manager)
		{
		}

		void LoadAssimpTextures(aiMaterial* assimpMaterial);
		void LoadAssimpConstants(aiMaterial* assimpMaterial);

	private:
		SharedMaterial m_material;
		MaterialManager* m_materialManager;
		bool m_isPBR = false;
	};


	class MaterialManager
	{
		friend class MaterialBuilder;

	public:
		MaterialManager() = default;
		MaterialBuilder CreateMaterialBuilder() { return MaterialBuilder(this); }

		SharedMaterial GetMaterial(std::wstring materialName);
		void AddMaterial(SharedMaterial material);
		void RemoveMaterial(SharedMaterial material);
		void LoadDefaultMaterials(TextureManager& textureManager);
		
		const std::wstring PHONG_DEFAULT = L"PhongDefault";
		const std::wstring PBR_DEFAULT = L"PBRDefault";

	private:

		
		std::mutex m_materialCacheMutex;
		std::unordered_map<std::wstring, SharedMaterial> m_materialCache;
	};
}