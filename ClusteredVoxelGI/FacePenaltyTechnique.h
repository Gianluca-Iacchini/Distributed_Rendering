#pragma once
#include "Technique.h"

namespace CVGI
{
	class FacePenaltyTechnique : public Technique
	{
	public:
		FacePenaltyTechnique(std::shared_ptr<TechniqueData> data);
		virtual ~FacePenaltyTechnique() {}

		void InitializeBuffers() override;

		void PerformTechnique(DX12Lib::ComputeContext& context) override;
		std::shared_ptr<DX12Lib::PipelineState> BuildPipelineState() override;

	protected:
		void TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize) override;
		std::shared_ptr<DX12Lib::RootSignature> BuildRootSignature() override;

	private:
		ConstantBufferIndirectLightTransport m_cbFacePenalty;

	public:
		static const std::wstring Name;

	private:
		enum class FacePenaltyRootSignature
		{
			VoxelCommonsCBV = 0,
			FacePenaltyCBV = 1,
			VoxelSRVTable,
			CompactSRVTable,
			ClusterVoxelSRVTable,
			FaceCountSRVTable,
			ClusterVisibilitySRVTable,
			FacePenaltyUAVTable,
			Count
		};
	};

}

