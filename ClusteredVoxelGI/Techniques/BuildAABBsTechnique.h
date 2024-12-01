#pragma once
#include "Technique.h"
#include "../Helpers/RayTracingHelpers.h"

namespace CVGI
{
	class BuildAABBsTechnique : public Technique
	{
	private:
		enum class AABBGenerationRootSignature
		{
			AABBGenerationCBV = 0,
			CompactSRVTable = 1,
			ClusterSRVTable,
			AABBGenerationUAVTable,
			Count
		};

	public:
		BuildAABBsTechnique(std::shared_ptr<TechniqueData> data)
		{
			m_bufferManager = std::make_shared<BufferManager>();
			data->SetBufferManager(Name, m_bufferManager);
			m_data = data;
		}
		~BuildAABBsTechnique() {}

		virtual void InitializeBuffers() override;
		virtual void PerformTechnique(DX12Lib::ComputeContext& context) override;
		virtual void TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize) override;

		virtual std::shared_ptr<DX12Lib::RootSignature> BuildRootSignature() override;
		virtual std::shared_ptr<DX12Lib::PipelineState> BuildPipelineState() override;

	public:
		static const std::wstring Name;
		ConstantBufferAABBGeneration m_cbAABBGeneration;

	private:
		enum class AABBBufferType
		{
			AABBData = 0,
			NextIndex = 1,
			Counter,
		};


	};
}



