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
		void InitializeGaussianConstants(DX12Lib::ComputeContext& context);

		std::shared_ptr<DX12Lib::PipelineState> BuildPipelineState() override;
		void SetIndirectCommandSignature(Microsoft::WRL::ComPtr<ID3D12CommandSignature> commandSignature) { m_commandSignature = commandSignature; }
		void CopyBufferData(DX12Lib::ComputeContext& context);
		void SwapBuffers();
		void SetGaussianBlock(UINT32 block);

		void PerformTechnique2(DX12Lib::ComputeContext& context);

	protected:
		void TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize);
		std::shared_ptr<DX12Lib::RootSignature> BuildRootSignature() override;

	private:
		std::shared_ptr<BufferManager> m_writeBufferManager;
		std::shared_ptr<BufferManager> m_readBufferManager;

		bool m_resetBuffers = true;

		ConstantBufferGaussianFilter m_cbGaussianFilter;
		Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_commandSignature;

	public:
		static const std::wstring Name;
		static const std::wstring ReadName;
		static const std::wstring WriteName;

	private:
		enum class GaussianFilterRootParameters
		{
			VoxelCommonCBV = 0,
			GaussianFilterCBV = 1,
			VoxelDataSRV,
			PrefixSumSRV,
			FacePenaltySRV,
			VoxelVisibleFaceSRV,
			RadianceBufferSRV,
			ReadBufferSRV,
			GaussianBufferUAV,
			WriteBufferUAV,
			Count
		};
	};
}


