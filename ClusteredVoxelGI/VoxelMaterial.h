#pragma once
#include "DX12Lib/Commons/Helpers.h"
#include "DX12Lib/Models/Material.h"

namespace CVGI
{
	class VoxelMaterial : public DX12Lib::Material
	{
	public:
		VoxelMaterial();
		virtual ~VoxelMaterial() = default;

		virtual DX12Lib::ConstantBufferMaterial BuildMaterialConstantBuffer() override;
		virtual void SetTransparent(bool isTransparent) override { m_isTransparent = false; }
		virtual void LoadDefaultTextures() override {}
		virtual UINT GetTextureCount() override { return 0; }
	};
}



