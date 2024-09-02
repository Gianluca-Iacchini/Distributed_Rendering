#pragma once
#include <DirectXMath.h>
#include "BufferManager.h"

namespace CVGI
{
	class ClusterVoxels
	{
	public:
		ClusterVoxels(DirectX::XMUINT3 VoxelSceneSize);
		~ClusterVoxels();

		void InitializeBuffers();
		void StartClustering();

	private:
		DirectX::XMUINT3 m_voxelSceneDimensions = DirectX::XMUINT3(128, 128, 128);
		BufferManager m_bufferManager;
	};
}

