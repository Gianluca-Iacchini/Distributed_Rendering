#include "Material.h"
#include "GraphicsCore.h"

using namespace Graphics;

void MaterialBuilder::AddTexture(aiTextureType assimpTextureType, aiString& texturePath)
{
	TextureType textureType = AssimpToTextureType(assimpTextureType);

	std::wstring texturePathW = Utils::ToWstring(texturePath.C_Str());
	SharedTexture texture = s_textureManager->LoadFromFile(texturePathW, false);

	AddTexture(textureType, texture);
}

void MaterialBuilder::AddTexture(TextureType textureType, SharedTexture texture)
{
	SharedTexture matTexture = texture;
	if (texture == nullptr)
	{
		matTexture = GetDefaultTextureForType(textureType);
	}

	m_material->m_textures[(UINT)textureType] = matTexture;
}

SharedTexture MaterialBuilder::GetDefaultTextureForType(TextureType textureType)
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
	case TextureType::BUMP_MAP:
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

	LoadAssimpTextures(assimpMaterial, textureHeap);
	LoadAssimpConstants(assimpMaterial);

	return Build(materialName, textureHeap);
}

SharedMaterial MaterialBuilder::Build(std::wstring& materialName, DescriptorHeap* textureHeap)
{
	m_material->m_name = materialName;

	for (UINT i = 0; i < (UINT)TextureType::NUM_TEXTURE_TYPES; ++i)
	{
		if (m_material->m_textures[i] == nullptr)
		{
			m_material->m_textures[i] = GetDefaultTextureForType((TextureType)i);
		}

		if (textureHeap != nullptr)
		{
			DescriptorHandle handle = textureHeap->Alloc(1);
			s_device->GetComPtr()->CopyDescriptorsSimple(1, handle, m_material->m_textures[i]->GetSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

			// Store the handle to be used in the root signature.
			// Technically we could just store the first handle, but for now we store all of them.
			m_material->m_textureSRVHandles[i] = handle;
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
		aiTextureType_NORMALS,
	};

	aiString texturePath;
	for (UINT i = 0; i < _countof(textureTypes); ++i)
	{
		aiTextureType textureType = textureTypes[i];

		if (assimpMaterial->GetTextureCount(textureType) > 0)
		{
			assimpMaterial->GetTexture(textureType, 0, &texturePath);

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
			AddTexture(TextureType::NORMAL_MAP, texture);
		else
			AddTexture(TextureType::BUMP_MAP, texture);
	}
	// Load default texture for height map.
	// We don't have to worry about normal map defaul texture since it is set in the loop above (if no normal map is found)
	else
	{
		AddTexture(TextureType::BUMP_MAP);
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

	if (assimpMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, diffuseColor) == aiReturn_SUCCESS)
		m_material->DiffuseColor = DirectX::XMFLOAT4(diffuseColor.r, diffuseColor.g, diffuseColor.b, diffuseColor.a);

	if (assimpMaterial->Get(AI_MATKEY_COLOR_SPECULAR, specularColor) == aiReturn_SUCCESS)
		m_material->SpecularColor = DirectX::XMFLOAT4(specularColor.r, specularColor.g, specularColor.b, specularColor.a);

	if (assimpMaterial->Get(AI_MATKEY_COLOR_AMBIENT, ambientColor) == aiReturn_SUCCESS)
		m_material->AmbientColor = DirectX::XMFLOAT4(ambientColor.r, ambientColor.g, ambientColor.b, ambientColor.a);

	if (assimpMaterial->Get(AI_MATKEY_COLOR_EMISSIVE, emissiveColor) == aiReturn_SUCCESS)
		m_material->EmissiveColor = DirectX::XMFLOAT4(emissiveColor.r, emissiveColor.g, emissiveColor.b, emissiveColor.a);

	if (assimpMaterial->Get(AI_MATKEY_OPACITY, opacity) == aiReturn_SUCCESS)
		m_material->Opacity = opacity;

	if (assimpMaterial->Get(AI_MATKEY_REFRACTI, indexOfRefraction) == aiReturn_SUCCESS)
		m_material->IndexOfRefraction = indexOfRefraction;

	if (assimpMaterial->Get(AI_MATKEY_SHININESS, shininess) == aiReturn_SUCCESS)
		m_material->Shininess = shininess;

	if (assimpMaterial->Get(AI_MATKEY_BUMPSCALING, bumpIntensity) == aiReturn_SUCCESS)
		m_material->BumpIntensity = bumpIntensity;
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

DirectX::GraphicsResource Material::CreateMaterialBuffer()
{
	return Graphics::s_graphicsMemory->AllocateConstant(CreateMaterialConstant());
}

void Material::UseMaterial(ID3D12GraphicsCommandList* cmdList)
{
	cmdList->SetGraphicsRootConstantBufferView(2, CreateMaterialBuffer().GpuAddress());

	// We only need to set the first texture since they are all contiguous in the heap.
	// The root signature knows how many are to be used.
	cmdList->SetGraphicsRootDescriptorTable(3, m_textureSRVHandles[0]);
}
