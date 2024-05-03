#include "Material.h"
#include "GraphicsCore.h"

using namespace Graphics;

void MaterialBuilder::AddTexture(aiTextureType assimpTextureType, aiString texturePath)
{
	TextureType textureType = GetDefaultTextureForType(assimpTextureType);

	SharedTexture texture = nullptr;
	if (texturePath != aiString(""))
	{
		std::wstring texturePathW = Utils::ToWstring(texturePath.C_Str());
		texture = s_textureManager->LoadFromFile(texturePathW, false);
	}

	AddTexture(textureType, texture);
}

void MaterialBuilder::AddTexture(TextureType textureType, SharedTexture texture)
{
	SharedTexture matTexture = texture;
	if (texture == nullptr)
	{
		matTexture = GetDefaultTexture(textureType);
	}

	m_material->m_textures[(UINT)textureType] = matTexture;
}

SharedTexture MaterialBuilder::GetDefaultTexture(TextureType textureType)
{
	switch (textureType)
	{
	case TextureType::DIFFUSE:
		return s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::WHITE_OPAQUE];
	case TextureType::SPECULAR:
		return s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::BLACK_OPAQUE];
	case TextureType::AMBIENT:
		return s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::WHITE_OPAQUE];
	case TextureType::EMISSIVE:
		return s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::BLACK_OPAQUE];
	case TextureType::NORMAL_MAP:
		return s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::NORMAL_MAP];
	default:
		return s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::MAGENTA];
	}
}

SharedMaterial MaterialBuilder::BuildFromAssimpMaterial(aiMaterial* assimpMaterial, DescriptorHeap* textureHeap)
{
	std::wstring materialName = Utils::ToWstring(assimpMaterial->GetName().C_Str());

	m_material = this->m_materialManager->GetMaterial(materialName);

	if (m_material != nullptr)
		return m_material;

	m_material = std::make_shared<Material>();


	aiString texturePath = aiString("");
	for (UINT i = 0; i < AI_TEXTURE_TYPE_MAX; ++i)
	{
		if (aiTextureType(i) == aiTextureType_DIFFUSE ||
			aiTextureType(i) == aiTextureType_SPECULAR ||
			aiTextureType(i) == aiTextureType_AMBIENT ||
			aiTextureType(i) == aiTextureType_EMISSIVE ||
			aiTextureType(i) == aiTextureType_NORMALS)
		{
			aiTextureType textureType = (aiTextureType)i;
			if (assimpMaterial->GetTextureCount(textureType) > 0)
				assimpMaterial->GetTexture(textureType, 0, &texturePath);
			else
				texturePath = aiString("");

			AddTexture(textureType, texturePath);
		}
	}

	return Build(materialName, textureHeap);
}

SharedMaterial MaterialBuilder::Build(std::wstring& materialName, DescriptorHeap* textureHeap)
{
	m_material->m_name = materialName;

	for (UINT i = 0; i < (UINT)TextureType::NUM_TEXTURE_TYPES; ++i)
	{
		if (m_material->m_textures[i] == nullptr)
		{
			m_material->m_textures[i] = GetDefaultTexture((TextureType)i);
		}

		if (textureHeap != nullptr)
		{
			if (i == 0)
			{
				DescriptorHandle handle = textureHeap->Alloc(1);
				s_device->GetComPtr()->CopyDescriptorsSimple(1, handle, m_material->m_textures[i]->GetSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
				m_material->m_textureSRVIndexStart = textureHeap->GetOffsetOfHandle(handle);
			}
		}
	}

	m_materialManager->AddMaterial(m_material);

	return m_material;
}

void MaterialManager::AddMaterial(SharedMaterial material)
{
	std::lock_guard<std::mutex> lock(m_materialCacheMutex);

	m_materialCache[material->m_name] = material;
}

void MaterialManager::RemoveMaterial(SharedMaterial material)
{
	std::lock_guard<std::mutex> lock(m_materialCacheMutex);

	m_materialCache.erase(material->m_name);
}

SharedMaterial MaterialManager::GetMaterial(std::wstring& materialName)
{
	std::lock_guard<std::mutex> lock(m_materialCacheMutex);

	auto it = m_materialCache.find(materialName);
	if (it != m_materialCache.end())
	{
		return it->second;
	}

	return nullptr;
}
