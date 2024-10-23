#pragma once
#include "Technique.h"

namespace CVGI
{
	class GaussianFilterTechnique : public Technique
	{
	public:
		GaussianFilterTechnique(std::shared_ptr<TechniqueData> data);
		void InitializeBuffers() override;
		void PerformTechnique(DX12Lib::ComputeContext& context) override;

		std::shared_ptr<DX12Lib::PipelineState> BuildPipelineState() override;
		void SetIndirectCommandSignature(Microsoft::WRL::ComPtr<ID3D12CommandSignature> commandSignature) { m_commandSignature = commandSignature; }

	protected:
		void TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize);
		std::shared_ptr<DX12Lib::RootSignature> BuildRootSignature() override;

	private:
		ConstantBufferGaussianFilter m_cbGaussianFilter;
		Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_commandSignature;

	public:
		static const std::wstring Name;

	private:
		enum class GaussianFilterRootParameters
		{
			VoxelCommonCBV = 0,
			GaussianFilterCBV = 1,
			VoxelDataSRV,
			PrefixSumSRV,
			FaceDataSRV,
			FacePenaltySRV,
			VoxelVisibleFaceSRV,
			VoxelRadianceUAV,
			Count
		};
	};
}


