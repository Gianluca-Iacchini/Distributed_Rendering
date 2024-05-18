#include "DX12Lib/pch.h"
#include "MaterialManager.h"

using namespace DX12Lib;
using namespace Graphics;

void MaterialBuilder::AddTexture(aiTextureType assimpTextureType, aiString& texturePath)
{
	MaterialTextureType textureType = AssimpToTextureType(assimpTextureType);

	std::wstring texturePathW = Utils::ToWstring(texturePath.C_Str());
	SharedTexture texture = Renderer::s_textureManager->LoadFromFile(texturePathW, false);

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
	case MaterialTextureType::BASECOLOR:
	case MaterialTextureType::AMBIENT:
	case MaterialTextureType::SHININESS:
	case MaterialTextureType::METALROUGHNESS:
	case MaterialTextureType::OPACITY:
		return Renderer::s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::WHITE_OPAQUE];
	case MaterialTextureType::SPECULAR:
	case MaterialTextureType::EMISSIVE:
		return Renderer::s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::BLACK_OPAQUE];
		return Renderer::s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::BLACK_OPAQUE];
	case MaterialTextureType::NORMAL_MAP:
	case MaterialTextureType::BUMP_MAP:
		return Renderer::s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::NORMAL_MAP];
	default:
		return Renderer::s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::MAGENTA];
	}
}

SharedMaterial MaterialBuilder::BuildFromAssimpMaterial(aiMaterial* assimpMaterial)
{
	std::wstring materialName = Utils::ToWstring(assimpMaterial->GetName().C_Str());

	m_material = this->m_materialManager->GetMaterial(materialName);

	if (m_material != nullptr)
		return m_material;

	m_isPBR = false;
	aiShadingMode shadingModel;
	if (assimpMaterial->Get(AI_MATKEY_SHADING_MODEL, shadingModel) == AI_SUCCESS)
	{
		if (shadingModel == aiShadingMode_PBR_BRDF)
			m_isPBR = true;
	}


	if (m_isPBR)
		m_material = std::make_shared<PBRMaterial>();
	else
		m_material = std::make_shared<PhongMaterial>();

	LoadAssimpTextures(assimpMaterial);
	LoadAssimpConstants(assimpMaterial);

	return Build(materialName);
}

SharedMaterial MaterialBuilder::Build(std::wstring& materialName)
{
	m_material->m_name = materialName;

	UINT numTextures = m_isPBR ? NUM_PBR_TEXTURES : NUM_PHONG_TEXTURES;

	for (UINT i = 0; i < numTextures; ++i)
	{
		if (m_material->m_textures[i] == nullptr)
		{
			MaterialTextureType textureType = (MaterialTextureType)i;
			m_material->SetTexture(textureType, GetDefaultTextureForType(textureType));
		}


		DescriptorHandle handle = Renderer::s_textureHeap->Alloc(1);
		s_device->GetComPtr()->CopyDescriptorsSimple(1, handle, m_material->m_textures[i]->GetSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		// Store only the first texture handle. The rest of the texture are placed after this one in the heap
		// The root signature already knows how many textures to use.			
		if (i == 0)
			m_material->m_firstTextureHandle = handle;
		
	}

	m_materialManager->AddMaterial(m_material);

	return m_material;
}

void MaterialBuilder::LoadAssimpTextures(aiMaterial* assimpMaterial)
{
	assert(assimpMaterial != nullptr);

	// Texture types to check.
	// Height map is checked separately since normal maps might used the height map slot.
	aiTextureType textureTypes[] =
	{
		// Common
		aiTextureType_EMISSIVE,
		aiTextureType_NORMALS,

		// PBR 
		aiTextureType_BASE_COLOR,
		aiTextureType_METALNESS,
		aiTextureType_DIFFUSE_ROUGHNESS,
		aiTextureType_AMBIENT_OCCLUSION,

		// Phong textures
		aiTextureType_DIFFUSE,
		aiTextureType_SPECULAR,
		aiTextureType_AMBIENT,
		aiTextureType_SHININESS,
		aiTextureType_OPACITY
	};

	aiString texturePath;

	for (UINT i = 0; i < _countof(textureTypes); ++i)
	{
		aiTextureType textureType = textureTypes[i];

		if (assimpMaterial->GetTextureCount(textureType) > 0)
		{
			if (assimpMaterial->GetTexture(textureType, 0, &texturePath) == AI_SUCCESS)
			{
				AddTexture(textureType, texturePath);

				if (textureType == aiTextureType_OPACITY)
					m_material->SetTransparent(true);
			}

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
		SharedTexture texture = Renderer::s_textureManager->LoadFromFile(texturePathW, false);

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
		return;
	}

	phongMat->SpecularColor = specColor;
}

void MaterialBuilder::SetAmbientColor(DirectX::XMFLOAT4 ambColor)
{
	PhongMaterial* phongMat = dynamic_cast<PhongMaterial*>(m_material.get());

	if (phongMat == nullptr)
	{
		return;
	}

	phongMat->AmbientColor = ambColor;
}

void MaterialBuilder::SetOpacity(float opacity)
{
	PhongMaterial* phongMat = dynamic_cast<PhongMaterial*>(m_material.get());

	if (phongMat == nullptr)
	{
		return;
	}


	phongMat->Opacity = opacity;

	if (opacity < 1)
		m_material->SetTransparent(true);
}

void MaterialBuilder::SetShininess(float shininess)
{
	PhongMaterial* phongMat = dynamic_cast<PhongMaterial*>(m_material.get());

	if (phongMat == nullptr)
	{
		return;
	}

	phongMat->Shininess = shininess;
}

void MaterialBuilder::SetIndexOfRefraction(float ior)
{
	PhongMaterial* phongMat = dynamic_cast<PhongMaterial*>(m_material.get());

	if (phongMat == nullptr)
	{
		return;
	}

	phongMat->IndexOfRefraction = ior;
}

void DX12Lib::MaterialBuilder::SetRoughness(float roughness)
{
	PBRMaterial* pbrMat = dynamic_cast<PBRMaterial*>(m_material.get());

	if (pbrMat == nullptr)
	{
		return;
	}

	pbrMat->Roughness = roughness;
}

void DX12Lib::MaterialBuilder::SetMetallic(float metallic)
{
	PBRMaterial* pbrMat = dynamic_cast<PBRMaterial*>(m_material.get());

	if (pbrMat == nullptr)
	{
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
		case aiTextureType_BASE_COLOR:
			return MaterialTextureType::BASECOLOR;
		case aiTextureType_METALNESS:
			return MaterialTextureType::METALROUGHNESS;
		case aiTextureType_DIFFUSE_ROUGHNESS:
			return MaterialTextureType::METALROUGHNESS;
		case aiTextureType_AMBIENT_OCCLUSION:
			return MaterialTextureType::OCCLUSION;
		case aiTextureType_OPACITY:
			return MaterialTextureType::OPACITY;
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
	aiString	transparency;

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

	if (assimpMaterial->Get(AI_MATKEY_BASE_COLOR, diffuseColor) == AI_SUCCESS)
		m_material->DiffuseColor = DirectX::XMFLOAT4(diffuseColor.r, diffuseColor.g, diffuseColor.b, diffuseColor.a);

	if (assimpMaterial->Get(AI_MATKEY_METALLIC_FACTOR, metallic) == AI_SUCCESS)
		this->SetMetallic(metallic);

	if (assimpMaterial->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughenss) == AI_SUCCESS)
		this->SetRoughness(roughenss);

	if (assimpMaterial->Get("$mat.gltf.alphaMode", 0, 0, transparency) == AI_SUCCESS)
		if (transparency != aiString("OPAQUE"))
			m_material->SetTransparent(true);
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