#pragma once

#include "Technique.h"

namespace CVGI
{
	class FaceCountTechnique : public Technique
	{
	public:
		FaceCountTechnique(std::shared_ptr<TechniqueData> data)
		{
			m_bufferManager = std::make_shared<BufferManager>();
			data->AddBufferManager(Name, m_bufferManager);
			m_data = data;
		}
		virtual ~FaceCountTechnique() {}

		virtual void InitializeBuffers() override;
		virtual void PerformTechnique(DX12Lib::ComputeContext& context) override;
		virtual void TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize) override;

		virtual std::shared_ptr<DX12Lib::PipelineState> BuildPipelineState() override;

	protected:
		virtual std::shared_ptr<DX12Lib::RootSignature> BuildRootSignature() override;

	public:
		static const std::wstring Name;

	private:
		UINT32 m_faceCount = 0;
		ConstantBufferFaceCount m_cbFaceCount;
	private:
		enum class FaceCountRootSignature
		{
			FaceCountCBV = 0,
			CompactSRVTable = 1,
			FaceCountUAVTable,
			Count
		};
	};
}