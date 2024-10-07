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

		bool IsBuilt() const { return m_built; }

	private:
		std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> m_geometryDescs;
		D3D12_RAYTRACING_INSTANCE_DESC m_instanceDesc;
		bool m_built = false;
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



