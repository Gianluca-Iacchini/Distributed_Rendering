#pragma once


#include "DirectXMath.h"
#include "DX12Lib/DXWrapper/Resource.h"
#include "DX12Lib/DXWrapper/GPUBuffer.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"

#include <vector>
#include <memory>
#include "d3dx12.h"
#include "DX12Lib/DXWrapper/PipelineState.h"
#include "DX12Lib/Commons/CommandContext.h"
#include "DX12Lib/Commons/MathHelper.h"

namespace CVGI
{


	struct BLASGeometry
	{
	public:
		BLASGeometry(UINT count, D3D12_GPU_VIRTUAL_ADDRESS startAddress, size_t stride);
		BLASGeometry(UINT count, D3D12_GPU_VIRTUAL_ADDRESS startAddress, size_t stride, DX12Lib::AABB& aabb);
		BLASGeometry(UINT count, D3D12_GPU_VIRTUAL_ADDRESS startAddress, size_t stride, DirectX::XMFLOAT3 min, DirectX::XMFLOAT3 max);
		BLASGeometry(DX12Lib::AABB& aabb) : m_geometryAABB(aabb) { m_geometryDesc = {}; }
		BLASGeometry(DirectX::XMFLOAT3 min, DirectX::XMFLOAT3 max) : m_geometryAABB({ min, max }) { m_geometryDesc = {}; }

		inline D3D12_RAYTRACING_GEOMETRY_DESC GetGeometryDesc() const { return m_geometryDesc; }
		inline void SetGeometryDesc(D3D12_RAYTRACING_GEOMETRY_DESC& desc) { m_geometryDesc = desc; }
		void SetGeometryDesc(UINT count, D3D12_GPU_VIRTUAL_ADDRESS startAddress, size_t stride);

		inline DX12Lib::AABB GetAABB() const { return m_geometryAABB; }
		inline void SetAABB(DX12Lib::AABB& aabb) { m_geometryAABB = aabb; }
		inline void SetAABB(DirectX::XMFLOAT3 min, DirectX::XMFLOAT3 max) { m_geometryAABB = { min, max }; }
	private:
		D3D12_RAYTRACING_GEOMETRY_DESC m_geometryDesc;
		DX12Lib::AABB m_geometryAABB;
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
		void AddGeometry(BLASGeometry& geometry);
		void AddGeometry(UINT count, D3D12_GPU_VIRTUAL_ADDRESS startAddress, size_t stride, DX12Lib::AABB geomAABB);
		virtual void Build(DX12Lib::CommandContext& context) override;

		D3D12_RAYTRACING_INSTANCE_DESC& GetInstanceDesc() { return m_instanceDesc; }

		bool IsBuilt() const { return m_built; }

	private:
		std::vector<BLASGeometry> m_geometries;
		D3D12_RAYTRACING_INSTANCE_DESC m_instanceDesc;
		bool m_built = false;
		DX12Lib::AABB m_blasAABB;
	};

	class TopLevelAccelerationStructure : public AccelerationStructure
	{
	public:
		virtual ~TopLevelAccelerationStructure() = default;
		void AddBLAS(std::shared_ptr<BottomLevelAccelerationStructure> structure);
		virtual void Build(DX12Lib::CommandContext& context) override;

		UINT GetBLASCount() const { return static_cast<UINT>(m_blasVector.size()); }

		std::vector<std::shared_ptr<BottomLevelAccelerationStructure>> m_blasVector;
	};

	
	class RayTracingContext : public DX12Lib::ComputeContext
	{
	public:
		virtual ~RayTracingContext() = default;

		static RayTracingContext& Begin();

		void DispatchRays1D(UINT width);
		void DispatchRays2D(UINT width, UINT height);
		void DispatchRays3D(UINT width, UINT height, UINT depth);
	};


}



