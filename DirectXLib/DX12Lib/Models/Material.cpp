#include "DX12Lib/pch.h"
#include "Material.h"

using namespace Graphics;
using namespace DX12Lib;



Material::Material()
{
	m_defaultPSO = PSO_PHONG_OPAQUE;
}


void Material::UseMaterial(ID3D12GraphicsCommandList* cmdList)
{
	// We only need to set the first texture since they are all contiguous in the heap.
	// The root signature knows how many are to be used.
	cmdList->SetGraphicsRootDescriptorTable((UINT)Renderer::RootSignatureSlot::MaterialTextureSRV, m_firstTextureHandle);
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
	if (m_textures == nullptr)
		return nullptr;

	if (index < (UINT)NUM_COMMON_TEXTURES)
		return m_textures[index].get();

	return nullptr;
}

void DX12Lib::Material::SetTransparent(bool isTransparent)
{
	m_isTransparent = isTransparent;

	if (m_isTransparent)
	{
		m_defaultPSO = PSO_PHONG_ALPHA_TEST;
	}
	else
	{
		m_defaultPSO = PSO_PHONG_OPAQUE;
	}
}

PhongMaterial::PhongMaterial()
{
	m_textures = new SharedTexture[NUM_PHONG_TEXTURES];
	m_defaultPSO = PSO_PHONG_OPAQUE;
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
	if (m_textures == nullptr)
		return nullptr;

	if (index < (UINT)NUM_PHONG_TEXTURES)
		return m_textures[index].get();

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

PBRMaterial::PBRMaterial()
{
	m_textures = new SharedTexture[NUM_PBR_TEXTURES];
	m_defaultPSO = PSO_PBR_OPAQUE;
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
	if (m_textures == nullptr)
		return nullptr;

	if (index < (UINT)NUM_PBR_TEXTURES)
		return m_textures[index].get();
	
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

	if (m_isTransparent)
	{
		m_defaultPSO = PSO_PBR_ALPHA_TEST;
	}
	else
	{
		m_defaultPSO = PSO_PBR_OPAQUE;
	}
}




