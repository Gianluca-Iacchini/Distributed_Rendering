#pragma once
#include "../Helpers/BufferManager.h"
#include "DirectXMath.h"
#include "Technique.h"


namespace CVGI
{
	class PrefixSumVoxels : public Technique
	{
	private:

		enum class PrefixSumBufferType
		{
			IndirectionRankBuffer,
			IndirectionIndexBuffer,
			CompactedVoxelIndex,
			CompactedHashedBuffer,
			PrefixSum,
		};

		enum class CompactBufferRootSignature
		{
			PrefixSumCBV = 0,
			VoxelizeUAVTable = 1,
			StreamCompactionUAVTable = 2,
			Count
		};

	public:
		PrefixSumVoxels(std::shared_ptr<TechniqueData> data);

		virtual ~PrefixSumVoxels() {}

		virtual void InitializeBuffers(DX12Lib::ComputeContext& context) override;

		virtual void PerformTechnique(DX12Lib::ComputeContext& context) override;
		virtual void TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize) override;
		void DeleteTemporaryBuffers();

		virtual std::shared_ptr<DX12Lib::RootSignature> BuildRootSignature() override;
		virtual std::shared_ptr<DX12Lib::PipelineState> BuildPipelineState() override;

	private:
		void ComputePrefixSumVariables();
	public:
		static const std::wstring Name;
	private:
		ConstantBufferCompactBuffer m_cbCompactBuffer;

		// Size of the prefix sum buffer for the current step
		std::vector<UINT32> v_prefixBufferSizeForStep;
		UINT64 m_prefixBufferSize = 0;
		bool m_firstSetIsSingleElement = false;

		UINT m_reduceStepCount = 0;
		UINT m_downSweepStepCount = 0;

		UINT m_currentStep = 0;
		UINT m_currentPhase = 0;

		const UINT ELEMENTS_PER_THREAD = 128;
	};
}

