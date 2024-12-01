#pragma once

#include "Technique.h"

namespace CVGI
{
	class LerpRadianceTechnique : public Technique
	{
	public:
		LerpRadianceTechnique(std::shared_ptr<TechniqueData> data);
		virtual ~LerpRadianceTechnique() = default;

		virtual void InitializeBuffers() override;
		virtual void PerformTechnique(DX12Lib::ComputeContext& context) override;

		void SetMaxTime(float maxTime);
		void SetAccumulatedTime(float accumulatedTime);
		void SetPhase(UINT phase);

		virtual std::shared_ptr<DX12Lib::PipelineState> BuildPipelineState() override;
	protected:
		virtual void TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize) override;
		virtual std::shared_ptr<DX12Lib::RootSignature> BuildRootSignature() override;

	public:
		static const std::wstring Name;

	private:
		ConstantBufferLerpRadiance m_cbLerpRadiance;

	private:
		enum class LerpRadianceRootSignature
		{
			VoxelCommonCBV = 0,
			LerpRadianceCBV = 1,
			GaussianFilterBufferUAV,
			LerpBufferUAV,
			Count
		};
	};
}



