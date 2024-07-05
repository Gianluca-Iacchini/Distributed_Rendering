#include "DX12Lib/pch.h"
#include "Material.h"

using namespace Graphics;
using namespace DX12Lib;



Material::Material()
{
	m_pso = Renderer::s_PSOs[PSO_PHONG_OPAQUE].get();
}

 
void DX12Lib::Material::SetTexture(UINT index, SharedTexture texture)
{
	if (m_textures != nullptr && index < GetTextureCount())
		m_textures[index] = texture;
}

Texture* DX12Lib::Material::GetTexture(MaterialTextureType type)
{
	return GetTexture((UINT)type);
}

Texture* DX12Lib::Material::GetTexture(UINT index)
{
	auto sharedTexture = GetSharedTexture(index);

	if (sharedTexture != nullptr)
		return sharedTexture.get();

	return nullptr;
}

SharedTexture DX12Lib::Material::GetSharedTexture(UINT index)
{
	if (m_textures == nullptr)
		return nullptr;

	if (index < (UINT)NUM_COMMON_TEXTURES)
		return m_textures[index];

	return nullptr;
}

void DX12Lib::Material::SetTransparent(bool isTransparent)
{
	m_isTransparent = isTransparent;

	std::wstring PSOName = L"PSO_PHONG_OPAQUE";

	if (m_isTransparent)
	{

		PSOName = PSO_PHONG_ALPHA_TEST;
	}
	
	m_pso = Renderer::s_PSOs[PSOName].get();
}

SharedTexture DX12Lib::Material::GetDefaultTextureForType(MaterialTextureType textureType)
{
	switch (textureType)
	{
	case MaterialTextureType::DIFFUSE:
	case MaterialTextureType::BASECOLOR:
		return Renderer::s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::MAGENTA];
	case MaterialTextureType::AMBIENT:
	case MaterialTextureType::SHININESS:
	case MaterialTextureType::OPACITY:
		return Renderer::s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::WHITE_OPAQUE];
	case MaterialTextureType::SPECULAR:
	case MaterialTextureType::EMISSIVE:
	case MaterialTextureType::OCCLUSION:
		return Renderer::s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::BLACK_OPAQUE];
	case MaterialTextureType::NORMAL_MAP:
	case MaterialTextureType::BUMP_MAP:
		return Renderer::s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::NORMAL_MAP];
	case MaterialTextureType::METALROUGHNESS:
		return Renderer::s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::RED_OPAQUE];
	default:
		return Renderer::s_textureManager->DefaultTextures[(UINT)TextureManager::DefaultTextures::MAGENTA];
	}
}

PhongMaterial::PhongMaterial()
{
	m_textures = new SharedTexture[NUM_PHONG_TEXTURES];
	m_pso = Renderer::s_PSOs[PSO_PHONG_OPAQUE].get();
}

void PhongMaterial::SetTexture(MaterialTextureType type, SharedTexture texture)
{
	if (m_textures != nullptr)
	{
		if (IS_PHONG(type))
		{
			m_textures[(UINT)type] = texture;
		}
	}
}

Texture* DX12Lib::PhongMaterial::GetTexture(UINT index)
{
	auto sharedTexture = GetSharedTexture(index);

	if (sharedTexture != nullptr)
		return sharedTexture.get();

	return nullptr;
}

SharedTexture DX12Lib::PhongMaterial::GetSharedTexture(UINT index)
{
	if (m_textures == nullptr)
		return nullptr;

	if (index < (UINT)NUM_PHONG_TEXTURES)
		return m_textures[index];

	return nullptr;
}

ConstantBufferMaterial DX12Lib::PhongMaterial::BuildMaterialConstantBuffer()
{
	ConstantBufferMaterial materialCB;

	materialCB.Float4_0 = DiffuseColor;
	materialCB.Float4_1 = EmissiveColor;
	materialCB.Float4_2 = SpecularColor;
	materialCB.Float4_3 = AmbientColor;
	materialCB.Float_0 = NormalScale;
	materialCB.Float_1 = Opacity;
	materialCB.Float_2 = Shininess;
	materialCB.Float_3 = IndexOfRefraction;

	return materialCB;
}

void DX12Lib::PhongMaterial::SetTransparent(bool isTransparent)
{
	Material::SetTransparent(isTransparent);
}

void DX12Lib::PhongMaterial::LoadDefaultTextures()
{
	if (m_textures == nullptr) return;

	for (UINT i = 0; i < NUM_PHONG_TEXTURES; i++)
	{
		if (m_textures[i] == nullptr)
			m_textures[i] = GetDefaultTextureForType((MaterialTextureType)i);
	}
}

PBRMaterial::PBRMaterial()
{
	m_textures = new SharedTexture[NUM_PBR_TEXTURES];
	m_pso = Renderer::s_PSOs[PSO_PBR_OPAQUE].get();
}

void PBRMaterial::SetTexture(MaterialTextureType type, SharedTexture texture)
{
	if (m_textures != nullptr)
	{
		if (IS_COMMON(type))
		{
			m_textures[(UINT)type] = texture;
		}


		else if (IS_PBR(type))
		{
			UINT offset = (UINT)type - PBR_TEXTURE_OFFSET;
			m_textures[NUM_COMMON_TEXTURES + offset] = texture;
		}
	}
}

Texture* DX12Lib::PBRMaterial::GetTexture(UINT index)
{
	auto sharedTex = GetSharedTexture(index);

	if (sharedTex != nullptr)
		return sharedTex.get();

	return nullptr;
}

SharedTexture DX12Lib::PBRMaterial::GetSharedTexture(UINT index)
{
	if (m_textures == nullptr)
		return nullptr;

	if (index < (UINT)NUM_PBR_TEXTURES)
		return m_textures[index];

	return nullptr;
}



ConstantBufferMaterial DX12Lib::PBRMaterial::BuildMaterialConstantBuffer()
{
	ConstantBufferMaterial materialCB;
	materialCB.Float4_0 = DiffuseColor;
	materialCB.Float4_1 = EmissiveColor;

	materialCB.Float_0 = NormalScale;
	materialCB.Float_1 = Metallic;
	materialCB.Float_2 = Roughness;

	return materialCB;
}

void DX12Lib::PBRMaterial::SetTransparent(bool isTransparent)
{
	m_isTransparent = isTransparent;

	std::wstring PSOName = L"PSO_PBR_OPAQUE";

	if (m_isTransparent)
	{
		PSOName = PSO_PBR_ALPHA_TEST;
	}

	m_pso = Renderer::s_PSOs[PSOName].get();
}

void DX12Lib::PBRMaterial::LoadDefaultTextures()
{
	if (m_textures == nullptr) return;

	for (UINT i = 0; i < NUM_PBR_TEXTURES; i++)
	{
		if (m_textures[i] != nullptr)
			continue;

		MaterialTextureType textureType = (MaterialTextureType)i;

		if (i >= NUM_COMMON_TEXTURES)
			textureType = (MaterialTextureType)(i + PBR_TEXTURE_OFFSET - NUM_COMMON_TEXTURES);

		m_textures[i] = GetDefaultTextureForType(textureType);
	}
}




