#pragma once

#include "Technique.h"
#include "array"

namespace VOX
{
	class LightTransportTechnique : public Technique
	{
		using Plane = DirectX::XMVECTOR;

	public:
		LightTransportTechnique(std::shared_ptr<TechniqueData> data, bool computeIndirect = true);
		virtual ~LightTransportTechnique() {}

		virtual void InitializeBuffers() override;
		virtual void PerformTechnique(DX12Lib::ComputeContext& context) override;
		virtual void BuildPipelineState() override;
	
		void BuildPipelineState(CD3DX12_SHADER_BYTECODE& shaderByteCode);

		void ClearRadianceBuffers(DX12Lib::ComputeContext& context, bool resetRadiance = false);
		void ComputeVisibleFaces(DX12Lib::ComputeContext& context);

		void ResetRadianceBuffers(bool reset);
		inline Microsoft::WRL::ComPtr<ID3D12CommandSignature> GetIndirectCommandSignature() const { return m_commandSignature; }

		void LaunchIndirectLightBlock(DX12Lib::ComputeContext& context, UINT blockCount);

		BufferManager* GetIndirectBufferManager() { return m_indirectBufferManager.get(); }

		void TransferRadianceData(DX12Lib::ComputeContext& context);

		std::uint8_t* GetVisibleFacesIndices(UINT32 visibleFacesCount);
		std::uint8_t* GetVisibleFacesRadiance(UINT32 visibleFacesCount);
		UINT32 GetVisibleFacesCount();

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
		bool m_computeIndirect = true;

		DX12Lib::ReadBackBuffer m_faceIndicesReadBack;
		DX12Lib::ReadBackBuffer m_faceRadianceReadback;
		DX12Lib::ReadBackBuffer m_visibleFacesCountReadback;

		std::uint8_t* m_visibleFacesIndices = nullptr;
		std::uint8_t* m_visibleFacesRadiance = nullptr;
		UINT32 m_visibleFacesCount = 0;

	private:
		ConstantBufferFrustumCulling m_cbFrustumCulling;
		ConstantBufferIndirectLightTransport m_cbLightIndirect;
		
		// Used for indirect dispatch
		Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_commandSignature;

		std::unique_ptr<DX12Lib::ComputePipelineState> m_indirectLightPso;

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
			OffsetCameraCBV,
			PrefixSumBuffersSRV,
			DepthMapSRV,
			LightTransportBuffersUAV,
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
			LitVoxelsSRV,
			LightTransportBuffersSRV,
			RadianceBufferUAV,
			Count
		};
	};

}


