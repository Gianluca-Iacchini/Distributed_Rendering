#include "DX12Lib/pch.h"
#include "Material.h"

using namespace Graphics;
using namespace DX12Lib;

#define IS_COMMON(tex) ((UINT)tex < NUM_COMMON_TEXTURES)
#define IS_PHONG(tex) ((UINT)tex < NUM_PHONG_TEXTURES)
#define IS_PBR(tex) (IS_COMMON(tex) || ((UINT)tex >= PBR_TEXTURE_OFFSET))

Material::Material()
{
	m_defaultPSO = PSO_PHONG_OPAQUE;
}


void Material::UseMaterial(ID3D12GraphicsCommandList* cmdList)
{
	// We only need to set the first texture since they are all contiguous in the heap.
	// The root signature knows how many are to be used.
	cmdList->SetGraphicsRootDescriptorTable(4, m_firstTextureHandle);
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
			m_textures[(UINT)type] = texture;

		else if (IS_PBR(type))
		{
			UINT offset = (UINT)type - PBR_TEXTURE_OFFSET;
			m_textures[NUM_COMMON_TEXTURES + offset] = texture;
		}
	}
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


