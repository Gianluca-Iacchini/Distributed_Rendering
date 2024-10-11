#pragma once

#include "Technique.h"

namespace CVGI
{
	class DisplayVoxelScene : public Technique
	{
	public:
		DisplayVoxelScene(std::shared_ptr<TechniqueData> data);
		virtual ~DisplayVoxelScene() override {} ;

		virtual void InitializeBuffers() override;
		virtual void PerformTechnique(DX12Lib::GraphicsContext& commandContext) override;

		void SetVertexData(DX12Lib::GraphicsContext& context);
		void SetCamera(DX12Lib::SceneCamera* camera);

		virtual std::shared_ptr<DX12Lib::RootSignature> BuildRootSignature() override;
		virtual std::shared_ptr<DX12Lib::PipelineState> BuildPipelineState() override;
	protected:
		virtual void TechniquePass(DX12Lib::GraphicsContext& commandContext) override;

	public:
		static const std::wstring Name;

	private:
		DX12Lib::TypedBuffer m_vertexBuffer;
		UINT64 m_vertexCount = 0;
		DX12Lib::SceneCamera* m_camera = nullptr;

	public:
		struct VertexSingleUINT
		{
			VertexSingleUINT() = default;

			VertexSingleUINT(const VertexSingleUINT&) = default;
			VertexSingleUINT& operator=(const VertexSingleUINT&) = default;

			VertexSingleUINT(VertexSingleUINT&&) = default;
			VertexSingleUINT& operator=(VertexSingleUINT&&) = default;

			VertexSingleUINT(UINT32 const& iposition) noexcept
				: position(iposition)
			{
			}

			VertexSingleUINT(UINT32 iposition) noexcept : position(iposition)
			{
			}

			UINT32 position;

			static const D3D12_INPUT_LAYOUT_DESC InputLayout;

		private:
			static constexpr unsigned int InputElementCount = 1;
			static const D3D12_INPUT_ELEMENT_DESC InputElements[InputElementCount];
		};

		enum class DisplayVoxelRootParameterSlot
		{
			VoxelCommonCBV = 0,
			CameraCBV = 1,
			VoxelSRVBufferTable = 2,
			CompactSRVBufferTable,
			ClusterSRVBufferTable,
			FaceVisibilitySRVBufferTable,
			ShadowSRVBufferTable,
			Count
		};
	};
}


