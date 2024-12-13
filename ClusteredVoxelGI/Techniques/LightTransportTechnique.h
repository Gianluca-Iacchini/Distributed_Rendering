#pragma once

#include "Technique.h"
#include "array"

namespace CVGI
{
	class LightTransportTechnique : public Technique
	{
		using Plane = DirectX::XMVECTOR;

	public:
		LightTransportTechnique(std::shared_ptr<TechniqueData> data);
		virtual ~LightTransportTechnique() {}

		void InitializeBuffers() override;
		void PerformTechnique(DX12Lib::ComputeContext& context) override;
		std::shared_ptr<DX12Lib::PipelineState> BuildPipelineState() override;
	
		void ResetRadianceBuffers(bool reset);
		inline Microsoft::WRL::ComPtr<ID3D12CommandSignature> GetIndirectCommandSignature() const { return m_commandSignature; }

		void LaunchIndirectLightBlock(DX12Lib::ComputeContext& context, UINT blockCount);

		BufferManager* GetIndirectBufferManager() { return m_indirectBufferManager.get(); }

	protected:
		void TechniquePass(DX12Lib::ComputeContext& commandContext, DirectX::XMUINT3 groupSize) override;
		void TechniquePassIndirect(DX12Lib::ComputeContext& commandContext);
		std::shared_ptr<DX12Lib::RootSignature> BuildRootSignature() override;
	private:
		Plane& TransformAndNormalize(Plane& plane);
		void CreateFrustumPlanes();
		void CreateExecuteIndirectCommandBuffer();
		std::shared_ptr<DX12Lib::RootSignature> BuildIndirectRootSignature();
		void BuildIndirectCommandPSO();


	public:
		static const std::wstring Name;
		static const std::wstring IndirectName;
	private:
		std::shared_ptr<BufferManager> m_indirectBufferManager;


	private:
		ConstantBufferFrustumCulling m_cbFrustumCulling;
		ConstantBufferIndirectLightTransport m_cbLightIndirect;
		
		// Used for indirect dispatch
		Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_commandSignature;

	public:
		enum class LightTransportBufferType
		{
			VisibleFaceCounter = 0,
			IndirectLightVisibleFacesIndices = 1,
			GaussianVisibleFacesIndices,
			IndirectLightUpdatedBitmap,
			GaussianUpdatedBitmap,
			GaussianDispatchBuffer,
			IndirectLightDispatchBuffer,
			Count
		};

	private:
		enum class LightTransportTechniqueRootParameters
		{
			VoxelCommonsCBV = 0,
			LightTransportCBV = 1,
			CameraCBV,
			PrefixSumBuffersSRV,
			AABBBuffersSRV,
			DepthMapSRV,
			//AccelerationStructureSRV,
			RadianceBufferUAV,
			LightTransportBuffersUAV,
			GaussianFilterBufferUAV,
			GaussianFinalWriteBufferUAV,
			Count
		};

		enum class LightTransportIndirectRootParameters
		{
			VoxelCommonsCBV = 0,
			IndirectCBV = 1,
			PrefixSumBuffersSRV,
			ClusterVoxelBufferSRV,
			AABBBuffersSRV,
			ClusterVisibilitySRV,
			FacePenaltyBufferSRV,
			LitVoxelsSRV,
			LightTransportBuffersSRV,
			RadianceBufferUAV,
			Count
		};
	};

}


