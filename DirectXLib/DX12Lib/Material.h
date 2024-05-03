#pragma once

#include "Helpers.h"
#include "Texture.h"
#include "assimp/material.h"

enum class TextureType
{
	DIFFUSE = 0,
	SPECULAR = 1,
	AMBIENT,
	EMISSIVE,
	NORMAL_MAP,
	BUMP_MAP,
	NUM_TEXTURE_TYPES
};

class MaterialManager;
class MaterialBuilder;

class Material
{
	friend class MaterialManager;
	friend class MaterialBuilder;

public:
	Material() = default;

	std::wstring& GetName() { return m_name; }
	UINT m_textureSRVIndexStart = 0;
private:
	std::wstring m_name;
	SharedTexture m_textures[(UINT)TextureType::NUM_TEXTURE_TYPES];

};

using SharedMaterial = std::shared_ptr<Material>;

class MaterialBuilder
{
	friend class MaterialManager;

public:

	void AddTexture(aiTextureType assimpTextureType, aiString& texturePath);
	void AddTexture(TextureType textureType, SharedTexture texture = nullptr);
	SharedTexture GetDefaultTextureForType(TextureType textureType);
	SharedMaterial BuildFromAssimpMaterial(aiMaterial* assimpMaterial, DescriptorHeap* textureHeap = nullptr);
	SharedMaterial Build(std::wstring& materialName, DescriptorHeap* textureHeap = nullptr);

	TextureType AssimpToTextureType(aiTextureType assimpTextureType)
	{
		switch (assimpTextureType)
		{
			case aiTextureType_DIFFUSE:
				return TextureType::DIFFUSE;
			case aiTextureType_SPECULAR:
				return TextureType::SPECULAR;
			case aiTextureType_AMBIENT:
				return TextureType::AMBIENT;
			case aiTextureType_EMISSIVE:
				return TextureType::EMISSIVE;
			case aiTextureType_NORMALS:
				return TextureType::NORMAL_MAP;
			case aiTextureType_HEIGHT:
				return TextureType::BUMP_MAP;
			default:
				return TextureType::DIFFUSE;
		}
	}



private:
	MaterialBuilder(MaterialManager* manager) : m_materialManager(manager) 
	{
		m_material = std::make_shared<Material>();
	}

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

	SharedMaterial GetMaterial(std::wstring& materialName);
	void AddMaterial(SharedMaterial material);
	void RemoveMaterial(SharedMaterial material);

private:
	void LoadDefaultMaterials(TextureManager& textureManager);

	std::mutex m_materialCacheMutex;
	std::unordered_map<std::wstring, SharedMaterial> m_materialCache;
};

