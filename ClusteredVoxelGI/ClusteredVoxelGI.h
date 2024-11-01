#pragma once
#include <DX12Lib/Commons/D3DApp.h>
#include <memory>
#include <DirectXMath.h>
#include "DX12Lib/DXWrapper/RootSignature.h"
#include "DX12Lib/DXWrapper/PipelineState.h"
#include "CVGIDataTypes.h"
#include "DX12Lib/DXWrapper/GPUBuffer.h"
#include "DX12Lib/DXWrapper/DescriptorHeap.h"
#include "GraphicsMemory.h"
#include "VoxelizeScene.h"
#include "DisplayVoxelScene.h"
#include "PrefixSumVoxels.h"
#include "ClusterVoxels.h"
#include "MergeClusters.h"
#include "ComputeNeighboursTechnique.h"
#include "ClusterVisibility.h"
#include "FaceCountTechnique.h"
#include "BuildAABBsTechnique.h"
#include "FacePenaltyTechnique.h"
#include "DX12Lib/Scene/LightComponent.h"
#include "Technique.h"
#include "LightTransportTechnique.h"
#include "GaussianFilterTechnique.h"
#include "LerpRadianceTechnique.h"

#include "LightVoxel.h"


namespace CVGI
{
	class VoxelCamera;

	class ClusteredVoxelGIApp : public DX12Lib::D3DApp
	{
	public:

		ClusteredVoxelGIApp(HINSTANCE hInstance, DX12Lib::Scene* scene)
			: D3DApp(hInstance, scene), m_vertexBuffer(DX12Lib::TypedBuffer(DXGI_FORMAT_R32_UINT))
		{

		}

		~ClusteredVoxelGIApp() = default;

		virtual void Initialize(DX12Lib::GraphicsContext& commandContext) override;
		virtual void Update(DX12Lib::GraphicsContext& commandContext) override;
		virtual void Draw(DX12Lib::GraphicsContext& commandContext) override;


	private:
		bool IsDirectXRaytracingSupported() const;

	public:
		const DirectX::XMFLOAT3 VoxelTextureDimension = DirectX::XMFLOAT3(512.0f, 512.0f, 512.0f);

	private:
		std::unique_ptr<VoxelizeScene> m_voxelizeScene;
		std::unique_ptr<DisplayVoxelScene> m_displayVoxelScene;
		std::unique_ptr<PrefixSumVoxels> m_prefixSumVoxels;
		std::unique_ptr<ClusterVoxels> m_clusterVoxels;
		std::unique_ptr<MergeClusters> m_mergeClusters;
		std::unique_ptr<ComputeNeighboursTechnique> m_computeNeighboursTechnique;
		std::unique_ptr<ClusterVisibility> m_clusterVisibility;
		std::unique_ptr<FaceCountTechnique> m_faceCountTechnique;
		std::unique_ptr<BuildAABBsTechnique> m_buildAABBsTechnique;
		std::unique_ptr<FacePenaltyTechnique> m_facePenaltyTechnique;
		std::unique_ptr<LightVoxel> m_lightVoxel;
		std::unique_ptr<LightTransportTechnique> m_lightTransportTechnique;
		std::unique_ptr<GaussianFilterTechnique> m_gaussianFilterTechnique;
		std::unique_ptr<LerpRadianceTechnique> m_lerpRadianceTechnique;

		std::unique_ptr<DX12Lib::Fence> m_rtgiFence;

		DirectX::GraphicsResource m_cbVoxelCommonsResource;

		DX12Lib::TypedBuffer m_vertexBuffer;

		D3D12_VIEWPORT m_voxelScreenViewport;
		D3D12_RECT m_voxelScissorRect;

		std::vector<DX12Lib::GPUBuffer*> m_uavBuffers;

		UINT64 m_numberOfVoxels = 0;

		DX12Lib::ShadowCamera* m_shadowCamera = nullptr;

		std::shared_ptr<TechniqueData> m_data = nullptr;

		float RTGIUpdateDelta = 0.0f;
	};
}