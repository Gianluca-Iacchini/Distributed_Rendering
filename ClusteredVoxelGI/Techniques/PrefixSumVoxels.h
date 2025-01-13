#pragma once

#include "DirectXMath.h"
#include "Technique.h"


namespace CVGI
{
	class PrefixSumVoxels : public VOX::Technique
	{
	private:
		enum class CompactBufferRootSignature
		{
			PrefixSumCBV = 0,
			VoxelizeUAVTable = 1,
			StreamCompactionUAVTable = 2,
			Count
		};

	public:
		enum class PrefixSumBufferType
		{
			IndirectionRankBuffer,
			IndirectionIndexBuffer,
			CompactedVoxelIndex,
			CompactedHashedBuffer,
			PrefixSum,
		};

	public:
		PrefixSumVoxels(std::shared_ptr<VOX::TechniqueData> data);

		virtual ~PrefixSumVoxels() {}

		virtual void InitializeBuffers(DX12Lib::ComputeContext& context) override;

		virtual void PerformTechnique(DX12Lib::ComputeContext& context) override;
		virtual void TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize) override;
		void DeleteTemporaryBuffers();

		virtual std::shared_ptr<DX12Lib::RootSignature> BuildRootSignature() override;
		virtual void BuildPipelineState() override;

		const std::vector<UINT32>& GetIndirectionRankBuffer() const;
		const std::vector<UINT32>& GetIndirectionIndexBuffer() const;
		const std::vector<UINT32>& GetCompactedVoxelIndexBuffer() const;
		const std::vector<UINT32>& GetCompactedHashedBuffer() const;

	private:
		void ComputePrefixSumVariables();
	public:
		static const std::wstring Name;
	private:
		::ConstantBufferCompactBuffer m_cbCompactBuffer;

		// Size of the prefix sum buffer for the current step
		std::vector<UINT32> v_prefixBufferSizeForStep;
		UINT64 m_prefixBufferSize = 0;
		bool m_firstSetIsSingleElement = false;

		UINT m_reduceStepCount = 0;
		UINT m_downSweepStepCount = 0;

		UINT m_currentStep = 0;
		UINT m_currentPhase = 0;

		const UINT ELEMENTS_PER_THREAD = 128;
		
		std::vector<UINT32> m_indRnkBuffer;
		std::vector<UINT32> m_indIdxBuffer;
		std::vector<UINT32> m_cmpIdxBuffer;
		std::vector<UINT32> m_cmpHshBuffer;
	};
}

