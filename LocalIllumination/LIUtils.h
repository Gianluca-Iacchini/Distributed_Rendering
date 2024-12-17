#pragma once

#include "DX12Lib/Commons/CommonConstants.h"
#include "../ClusteredVoxelGI/Shaders/TechniquesCompat.h"

namespace LI
{
	class LIUtils
	{
	public:
		static ConstantBufferVoxelCommons BuildVoxelCommons(DX12Lib::AABB SceneAABB, DirectX::XMUINT3 VoxelGridSize);
		
	};
}



