#pragma once
#include "Technique.h"

namespace VOX
{
	class GaussianFilterTechnique : public Technique
	{
	public:
		GaussianFilterTechnique(std::shared_ptr<TechniqueData> data);
		void InitializeBuffers() override;
		void PerformTechnique(DX12Lib::ComputeContext& context) override;
		void InitializeGaussianConstants(DX12Lib::ComputeContext& context);

		virtual void BuildPipelineState() override;
		void SetIndirectCommandSignature(Microsoft::WRL::ComPtr<ID3D12CommandSignature> commandSignature) { m_commandSignature = commandSignature; }

		void SetGaussianBlock(UINT32 block);

		void SetGaussianKernelSize(UINT size);
		UINT GetGaussianKernelSize() { return m_cbGaussianFilter.KernelSize; }

		bool GetUsePrecomputedGaussian() { return (bool)m_cbGaussianFilter.UsePreComputedGaussian; }
		void SetUsePrecomputedGaussian(bool usePrecomp);

		UINT GetGaussianPassCount() { return m_cbGaussianFilter.PassCount; }
		void SetGaussianPassCount(UINT passCount);

		float GetGaussianSigma() { return m_cbGaussianFilter.Sigma; }
		void SetGaussianSigma(float sigma);

		bool GaussianOptionModified() { return m_gaussianOptionModified; }

		void PerformTechnique2(DX12Lib::ComputeContext& context);
		void TransferRadianceData(DX12Lib::ComputeContext& context);

		std::vector<DirectX::XMUINT2>& GetRadianceData();
		std::uint8_t* GetRadianceDataPtr();
		UINT GetRadianceLength() { return m_radLength; }

		std::vector<DirectX::XMUINT2> m_radianceData;
		UINT m_radLength = 0;

		virtual UINT64 GetMemoryUsage() override;

	protected:
		void TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize);
		std::shared_ptr<DX12Lib::RootSignature> BuildRootSignature() override;

	private:
		std::shared_ptr<BufferManager> m_readBufferManager;

		bool m_resetBuffers = true;

		ConstantBufferGaussianFilter m_cbGaussianFilter;
		Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_commandSignature;

		DX12Lib::ReadBackBuffer m_radianceReadBack;

		bool m_gaussianOptionModified = false;

	public:
		static const std::wstring Name;
		static const std::wstring ReadName;

	private:
		enum class GaussianFilterRootParameters
		{
			VoxelCommonCBV = 0,
			GaussianFilterCBV = 1,
			VoxelDataSRV,
			PrefixSumSRV,
			VoxelVisibleFaceSRV,
			RadianceBufferSRV,
			GaussianBufferUAV,
			WriteBufferUAV,
			Count
		};
	};
}


