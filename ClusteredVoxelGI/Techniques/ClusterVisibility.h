#pragma once

#include "../Helpers/BufferManager.h"
#include "../Helpers/RayTracingHelpers.h"
#include "../Helpers/RaytracingStateObject.h"
#include "DX12Lib/Commons/MathHelper.h"
#include "../Shaders/TechniquesCompat.h"
#include "Technique.h"

namespace DX12Lib
{
	class ComputeContext;
	class RootSignature;
	class ComputePipelineState;
	class SceneCamera;
}

namespace CVGI
{

	class ClusterVisibility : public Technique
	{
	public:
		ClusterVisibility(std::shared_ptr<TechniqueData> data)
		{
			m_bufferManager = std::make_shared<BufferManager>();
			data->SetBufferManager(Name, m_bufferManager);
			m_data = data;
		}

		virtual ~ClusterVisibility() {}

		virtual void InitializeBuffers() override;
		virtual void PerformTechnique(DX12Lib::ComputeContext& context) override;
		virtual void TechniquePass(RayTracingContext& context, DirectX::XMUINT3 groupSize) override;

		virtual std::shared_ptr<DX12Lib::RootSignature> BuildRootSignature() override;
		virtual std::shared_ptr<DX12Lib::PipelineState> BuildPipelineState() override;

		std::unique_ptr<TopLevelAccelerationStructure> BuildAccelerationStructures(DX12Lib::ComputeContext& context);

	public:
		static const std::wstring Name;

	private:
		RTSceneVisibility m_cbRayTracing;
	private:


		enum class RayTraceRootSignature
		{
			RayTraceCBV = 0,
			AccelerationStructureSRV = 1,
			VoxelSRVTable,
			CompactSRVTable,
			ClusterSRVTable,
			AABBSRVTable,
			RayTraceUAVTable,
			Count
		};

		enum class BufferType
		{
			FaceClusterVisibility = 0,
			VisibleCluster = 1,
			GeometryOffset,
			AABBOffset,
			ClusterCount,
			Count
		};
		
	};

}

