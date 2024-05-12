#include "DX12Lib/pch.h"
#include "Material.h"

using namespace Graphics;
using namespace DX12Lib;

#define IS_COMMON(tex) ((UINT)tex < NUM_COMMON_TEXTURES)
#define IS_PHONG(tex) ((UINT)tex < NUM_PHONG_TEXTURES)
#define IS_PBR(tex) (IS_COMMON(tex) || ((UINT)tex >= PBR_TEXTURE_OFFSET))


void Material::UseMaterial(ID3D12GraphicsCommandList* cmdList)
{
	cmdList->SetGraphicsRootConstantBufferView(2, CreateMaterialBuffer().GpuAddress());

	// We only need to set the first texture since they are all contiguous in the heap.
	// The root signature knows how many are to be used.
	cmdList->SetGraphicsRootDescriptorTable(4, m_firstTextureHandle);
}

ConstantBufferPhongMaterial DX12Lib::PhongMaterial::CreatePhongMaterialBuffer()
{
	ConstantBufferPhongMaterial cb;
	cb.DiffuseColor = DiffuseColor;
	cb.SpecularColor = SpecularColor;
	cb.AmbientColor = AmbientColor;
	cb.EmissiveColor = EmissiveColor;

	cb.NormalScale = NormalScale;
	cb.Opacity = Opacity;
	cb.Shininess = Shininess;
	cb.IndexOfRefraction = IndexOfRefraction;

	return cb;
}

DirectX::GraphicsResource PhongMaterial::CreateMaterialBuffer()
{
	ConstantBufferPhongMaterial cb;
	cb.DiffuseColor = DiffuseColor;
	cb.SpecularColor = SpecularColor;
	cb.AmbientColor = AmbientColor;
	cb.EmissiveColor = EmissiveColor;

	cb.NormalScale = NormalScale;
	cb.Opacity = Opacity;
	cb.Shininess = Shininess;
	cb.IndexOfRefraction = IndexOfRefraction;

	return Graphics::s_graphicsMemory->AllocateConstant<ConstantBufferPhongMaterial>(cb);
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

ConstantBufferPBRMaterial DX12Lib::PBRMaterial::CreatePBRMaterialBuffer()
{
	ConstantBufferPBRMaterial cb;
	cb.BaseColor = DiffuseColor;
	cb.EmissiveColor = EmissiveColor;

	cb.NormalScale = NormalScale;
	cb.Metallic = Metallic;
	cb.Roughness = Roughness;

	return cb;
}

DirectX::GraphicsResource PBRMaterial::CreateMaterialBuffer()
{
	ConstantBufferPBRMaterial cb;
	cb.BaseColor = DiffuseColor;
	cb.EmissiveColor = EmissiveColor;
	
	cb.NormalScale = NormalScale;
	cb.Metallic = Metallic;
	cb.Roughness = Roughness;

	return Graphics::s_graphicsMemory->AllocateConstant<ConstantBufferPBRMaterial>(cb);
}

