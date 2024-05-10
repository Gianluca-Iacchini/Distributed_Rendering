#include "pch.h"
#include "MaterialManager.h"

using namespace DX12Lib;
using namespace Graphics;

void MaterialBuilder::AddTexture(aiTextureType assimpTextureType, aiString& texturePath)
{
	MaterialTextureType textureType = AssimpToTextureType(assimpTextureType);

	std::wstring texturePathW = Utils::ToWstring(texturePath.C_Str());
	SharedTexture texture = s_textureManager->LoadFromFile(texturePathW, false);

	AddTexture(textureType, texture);
}

void MaterialBuilder::AddTexture(MaterialTextureType textureType, SharedTexture texture)
{
	SharedTexture matTexture = texture;
	if (texture == nullptr)
	{
		matTexture = GetDefaultTextureForType(textureType);
	}

	m_material->SetTexture(textureType, matTexture);
}

SharedTexture MaterialBuilder::GetDefaultTextureForType(MaterialTextureType textureType)
{
	switch (textureType)
	{
	case MaterialTextureType::DIFFUSE:
		return s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::WHITE_OPAQUE];
	case MaterialTextureType::SPECULAR:
		return s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::BLACK_OPAQUE];
	case MaterialTextureType::AMBIENT:
		return s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::WHITE_OPAQUE];
	case MaterialTextureType::EMISSIVE:
		return s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::BLACK_OPAQUE];
	case MaterialTextureType::SHININESS:
		return s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::WHITE_OPAQUE];
	case MaterialTextureType::NORMAL_MAP:
	case MaterialTextureType::BUMP_MAP:
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

	if (m_isPBR)
		m_material = std::make_shared<PBRMaterial>();
	else
		m_material = std::make_shared<PhongMaterial>();

	LoadAssimpTextures(assimpMaterial, textureHeap);
	LoadAssimpConstants(assimpMaterial);

	return Build(materialName, textureHeap);
}

SharedMaterial MaterialBuilder::Build(std::wstring& materialName, DescriptorHeap* textureHeap)
{
	m_material->m_name = materialName;

	for (UINT i = 0; i < NUM_PHONG_TEXTURES; ++i)
	{
		if (m_material->m_textures[i] == nullptr)
		{
			MaterialTextureType textureType = (MaterialTextureType)i;
			m_material->SetTexture(textureType, GetDefaultTextureForType(textureType));
		}

		if (textureHeap != nullptr)
		{
			DescriptorHandle handle = textureHeap->Alloc(1);
			s_device->GetComPtr()->CopyDescriptorsSimple(1, handle, m_material->m_textures[i]->GetSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

			// Store only the first texture handle. The rest of the texture are placed after this one in the heap
			// The root signature already knows how many textures to use.			
			if (i == 0)
				m_material->m_firstTextureHandle = handle;
		}
	}

	m_materialManager->AddMaterial(m_material);

	return m_material;
}

void MaterialBuilder::LoadAssimpTextures(aiMaterial* assimpMaterial, DescriptorHeap* textureHeap)
{
	assert(assimpMaterial != nullptr);

	// Texture types to check.
	// Height map is checked separately since normal maps might used the height map slot.
	aiTextureType textureTypes[] =
	{
		aiTextureType_DIFFUSE,
		aiTextureType_SPECULAR,
		aiTextureType_AMBIENT,
		aiTextureType_EMISSIVE,
		aiTextureType_SHININESS,
		aiTextureType_NORMALS,
	};

	aiString texturePath;

	for (UINT i = 0; i < _countof(textureTypes); ++i)
	{
		aiTextureType textureType = textureTypes[i];

		if (assimpMaterial->GetTextureCount(textureType) > 0)
		{
			if (assimpMaterial->GetTexture(textureType, 0, &texturePath) == AI_SUCCESS)
				AddTexture(textureType, texturePath);
		}
		// Load default texture
		else
		{
			AddTexture(AssimpToTextureType(textureType));
		}
	}

	// Materials might store normal map in height map slot
	if (assimpMaterial->GetTextureCount(aiTextureType_HEIGHT) > 0)
	{
		assimpMaterial->GetTexture(aiTextureType_HEIGHT, 0, &texturePath);

		std::wstring texturePathW = Utils::ToWstring(texturePath.C_Str());
		SharedTexture texture = s_textureManager->LoadFromFile(texturePathW, false);

		// Height map usually has 8 bits per pixel, while normal map usuall has 24.
		// If for some reason a material has a normal map and a normal map in the height map slot then
		// the normal map will be replaced by the one in the height map slot.
		// (If the model has both a normal map and a height map then both will be initiated in the appropiate slots)
		if (DirectX::BitsPerPixel(texture->GetDesc().Format) >= 24)
			AddTexture(MaterialTextureType::NORMAL_MAP, texture);
		else
			AddTexture(MaterialTextureType::BUMP_MAP, texture);
	}
	// Load default texture for height map.
	// We don't have to worry about normal map defaul texture since it is set in the loop above (if no normal map is found)
	else
	{
		AddTexture(MaterialTextureType::BUMP_MAP);
	}
}


void MaterialBuilder::SetSpecularColor(DirectX::XMFLOAT4 specColor)
{
	PhongMaterial* phongMat = dynamic_cast<PhongMaterial*>(m_material.get());

	if (phongMat == nullptr)
	{
		DXLIB_CORE_WARN("Trying to set specular color on a non phong material");
		return;
	}

	phongMat->SpecularColor = specColor;
}

void MaterialBuilder::SetAmbientColor(DirectX::XMFLOAT4 ambColor)
{
	PhongMaterial* phongMat = dynamic_cast<PhongMaterial*>(m_material.get());

	if (phongMat == nullptr)
	{
		DXLIB_CORE_WARN("Trying to set ambient color on a non phong material");
		return;
	}

	phongMat->AmbientColor = ambColor;
}

void MaterialBuilder::SetOpacity(float opacity)
{
	PhongMaterial* phongMat = dynamic_cast<PhongMaterial*>(m_material.get());

	if (phongMat == nullptr)
	{
		DXLIB_CORE_WARN("Trying to set opacity on a non phong material");
		return;
	}

	phongMat->Opacity = opacity;

}

void MaterialBuilder::SetShininess(float shininess)
{
	PhongMaterial* phongMat = dynamic_cast<PhongMaterial*>(m_material.get());

	if (phongMat == nullptr)
	{
		DXLIB_CORE_WARN("Trying to set shininess on a non phong material");
		return;
	}

	phongMat->Shininess = shininess;
}

void MaterialBuilder::SetIndexOfRefraction(float ior)
{
	PhongMaterial* phongMat = dynamic_cast<PhongMaterial*>(m_material.get());

	if (phongMat == nullptr)
	{
		DXLIB_CORE_WARN("Trying to set index of refraction on a non phong material");
		return;
	}

	phongMat->IndexOfRefraction = ior;
}

void DX12Lib::MaterialBuilder::SetRoughness(float roughness)
{
	PBRMaterial* pbrMat = dynamic_cast<PBRMaterial*>(m_material.get());

	if (pbrMat == nullptr)
	{
		DXLIB_CORE_WARN("Trying to set index of refraction on a non phong material");
		return;
	}

	pbrMat->Roughness = roughness;
}

void DX12Lib::MaterialBuilder::SetMetallic(float metallic)
{
	PBRMaterial* pbrMat = dynamic_cast<PBRMaterial*>(m_material.get());

	if (pbrMat == nullptr)
	{
		DXLIB_CORE_WARN("Trying to set index of refraction on a non phong material");
		return;
	}

	pbrMat->Metallic = metallic;
}

MaterialTextureType DX12Lib::MaterialBuilder::AssimpToTextureType(aiTextureType assimpTextureType)
{
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
}

void MaterialBuilder::LoadAssimpConstants(aiMaterial* assimpMaterial)
{
	assert(assimpMaterial != nullptr);

	aiColor4D   diffuseColor;
	aiColor4D   specularColor;
	aiColor4D   ambientColor;
	aiColor4D   emissiveColor;
	float       opacity;
	float       indexOfRefraction;
	float       shininess;
	float       bumpIntensity;
	float		metallic;
	float		roughenss;

	if (assimpMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, diffuseColor) == aiReturn_SUCCESS)
		m_material->DiffuseColor = DirectX::XMFLOAT4(diffuseColor.r, diffuseColor.g, diffuseColor.b, diffuseColor.a);

	if (assimpMaterial->Get(AI_MATKEY_COLOR_SPECULAR, specularColor) == aiReturn_SUCCESS)
		this->SetSpecularColor(DirectX::XMFLOAT4(specularColor.r, specularColor.g, specularColor.b, specularColor.a));

	if (assimpMaterial->Get(AI_MATKEY_COLOR_AMBIENT, ambientColor) == aiReturn_SUCCESS)
		this->SetAmbientColor(DirectX::XMFLOAT4(ambientColor.r, ambientColor.g, ambientColor.b, ambientColor.a));

	if (assimpMaterial->Get(AI_MATKEY_COLOR_EMISSIVE, emissiveColor) == aiReturn_SUCCESS)
		m_material->EmissiveColor = DirectX::XMFLOAT4(emissiveColor.r, emissiveColor.g, emissiveColor.b, emissiveColor.a);

	if (assimpMaterial->Get(AI_MATKEY_OPACITY, opacity) == aiReturn_SUCCESS)
		this->SetOpacity(opacity);

	if (assimpMaterial->Get(AI_MATKEY_REFRACTI, indexOfRefraction) == aiReturn_SUCCESS)
		this->SetIndexOfRefraction(indexOfRefraction);

	if (assimpMaterial->Get(AI_MATKEY_SHININESS, shininess) == aiReturn_SUCCESS)
		this->SetShininess(shininess);

	if (assimpMaterial->Get(AI_MATKEY_BUMPSCALING, bumpIntensity) == aiReturn_SUCCESS)
		this->SetNormalScale(bumpIntensity);

	if (assimpMaterial->Get(AI_MATKEY_METALLIC_FACTOR, metallic) == AI_SUCCESS)
		this->SetMetallic(metallic);

	if (assimpMaterial->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughenss) == AI_SUCCESS)
		this->SetRoughness(roughenss);
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