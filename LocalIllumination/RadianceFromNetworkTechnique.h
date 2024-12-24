#pragma once

#include "Technique.h"

namespace LI
{

	class RadianceFromNetworkTechnique : public VOX::Technique
	{
	public:
		RadianceFromNetworkTechnique(std::shared_ptr<VOX::TechniqueData> data);
		virtual ~RadianceFromNetworkTechnique() = default;

		virtual void InitializeBuffers() override;


		virtual void BuildPipelineState() override;

		UINT64 ProcessNetworkData(DX12Lib::ComputeContext& context, DX12Lib::UploadBuffer* buffer, UINT faceCount, UINT shouldReset);

	protected:
		virtual void PerformTechnique(DX12Lib::ComputeContext& context) override;
		virtual void TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize) override;

	protected:
		virtual std::shared_ptr<DX12Lib::RootSignature> BuildRootSignature() override;
	private:
		ConstantBufferRadianceFromNetwork m_cbRadianceFromNetwork;
	};
}

