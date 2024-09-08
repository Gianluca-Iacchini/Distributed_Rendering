#pragma once


#include "DirectXMath.h"
#include "DX12Lib/DXWrapper/Resource.h"
#include "DX12Lib/DXWrapper/GPUBuffer.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"
#include <vector>
#include <memory>
namespace DX12Lib
{
	class CommandContext;
	class ComputeContext;
}

namespace CVGI
{


	struct AABB
	{
		DirectX::XMFLOAT3 min;
		DirectX::XMFLOAT3 max;
	};


	class AccelerationStructure : public DX12Lib::GPUBuffer
	{
	public:
		virtual ~AccelerationStructure() = default;
		virtual void Build(DX12Lib::CommandContext& context) = 0;

	protected:
		AccelerationStructure() = default;
		virtual void Create(UINT byteSize);
		virtual void Create(UINT32 elementCount, UINT32 elementSize) override;
		virtual void CreateDerivedViews() override;
	};

	class BottomLevelAccelerationStructure : public AccelerationStructure
	{
	public:
		virtual ~BottomLevelAccelerationStructure() = default;
		void AddGeometry(UINT count, D3D12_GPU_VIRTUAL_ADDRESS startAddress, size_t stride);
		virtual void Build(DX12Lib::CommandContext& context) override;

		D3D12_RAYTRACING_INSTANCE_DESC& GetInstanceDesc() { return m_instanceDesc; }

	private:
		std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> m_geometryDescs;
		D3D12_RAYTRACING_INSTANCE_DESC m_instanceDesc;
	};

	class TopLevelAccelerationStructure : public AccelerationStructure
	{
	public:
		virtual ~TopLevelAccelerationStructure() = default;
		void AddBLAS(std::shared_ptr<BottomLevelAccelerationStructure> structure);
		virtual void Build(DX12Lib::CommandContext& context) override;

		std::vector<std::shared_ptr<BottomLevelAccelerationStructure>> m_blasVector;
	};

	class Octree
	{
	public:
		Octree(std::vector<AABB> elementsAABBs, DirectX::XMFLOAT3 octreeSize) : 
			m_elementsAABBs(elementsAABBs), m_OctreeSize(octreeSize) {}
		~Octree() {}

		void CreateOctree(unsigned int maxDepth = 10, unsigned int maxElements = 10);
		std::vector<std::vector<unsigned int>> GetLeaves();

	private:
		void CreateOctreeRecursive(DirectX::XMFLOAT3 minBoundary, DirectX::XMFLOAT3 maxBoundary, std::vector<unsigned int> currentIndices, unsigned int currentDepth);

	private:
		std::vector<AABB> m_elementsAABBs;
		DirectX::XMFLOAT3 m_OctreeSize;
		std::vector<std::vector<unsigned int>> m_elementsPerLeaf;

		unsigned int m_maxDepth = 10;
		unsigned int m_maxElements = 10;
	};


}



