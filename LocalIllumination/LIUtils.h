#pragma once

#include "DX12Lib/Commons/CommonConstants.h"

namespace LI
{
	class LIUtils
	{
	public:
		static DX12Lib::ConstantBufferVoxelTransform BuildVoxelCommons(DX12Lib::AABB SceneAABB, DirectX::XMUINT3 VoxelGridSize);

	};
}



