#pragma once

#include "Material.h"

namespace DX12Lib
{
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



	private:
		MaterialBuilder(MaterialManager* manager, bool isPBR = false)
			: m_materialManager(manager), m_isPBR(isPBR)
		{
		}

		void LoadAssimpTextures(aiMaterial* assimpMaterial, DescriptorHeap* textureHeap);
		void LoadAssimpConstants(aiMaterial* assimpMaterial);

	private:
		bool m_isPBR = false;
		SharedMaterial m_material;
		MaterialManager* m_materialManager;
	};


	class MaterialManager
	{
		friend class MaterialBuilder;

	public:
		MaterialManager() = default;
		//SharedMaterial LoadMaterial(const )
		MaterialBuilder CreateMaterialBuilder(bool isPBR = false) { return MaterialBuilder(this, isPBR); }

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