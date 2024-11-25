#pragma once

#include <memory>
#include "BufferManager.h"
#include "Shaders/TechniquesCompat.h"
#include <unordered_map>
#include "RayTracingHelpers.h"
#include "WinPixEventRuntime/pix3.h"
#include "DX12Lib/Commons/MathHelper.h"
#include "GraphicsMemory.h"
#include "DX12Lib/DXWrapper/DescriptorHeap.h"
#include "DX12Lib/Commons/CommonConstants.h"

namespace DX12Lib
{
	class RootSignature;
	class PipelineState;
	class GraphicsContext;
	class ComputeContext;
	class SceneCamera;
	class LightComponent;
}

namespace CVGI
{
	class RayTracingContext;

	class TechniqueData
	{
	public:
		TechniqueData() { };
		~TechniqueData() {};

		BufferManager& GetBufferManager(std::wstring name);

		void SetBufferManager(std::wstring name, std::shared_ptr<BufferManager> bufferManager)
		{
			m_bufferManagers[name] = bufferManager;
		}

		UINT32 GetVoxelCount() { return m_voxelCount; }
		UINT32 GetClusterCount() { return m_clusterCount; }

		void SetVoxelCount(UINT32 count);
		void SetClusterCount(UINT32 count);

		void SetCamera(DX12Lib::SceneCamera* camera);
		DX12Lib::SceneCamera* GetCamera() const { return m_sceneCamera; }

		void SetLightComponent(DX12Lib::LightComponent* lightComponent) { m_lightComponent = lightComponent; }
		DX12Lib::LightComponent* GetLightComponent() const { return m_lightComponent; }

		void SetTlas(std::unique_ptr<TopLevelAccelerationStructure>&& tlas) { m_tlas = std::move(tlas); }
		const TopLevelAccelerationStructure* GetTlas() { return m_tlas.get(); }

		DirectX::XMUINT3 GetVoxelGridSize() { return VoxelGridSize; }
		DirectX::XMFLOAT3 GetVoxelCellSize() { return VoxelCellSize; }

		void SetVoxelGridSize(DirectX::XMUINT3 size);
		void SetVoxelCellSize(DirectX::XMFLOAT3 size);

		void SetSceneAABB(DX12Lib::AABB aabb);
		DX12Lib::AABB GetSceneAABB() { return SceneAABB; }

		const DirectX::XMFLOAT4X4& GetVoxelToWorldMatrix() const { return m_cbVoxelCommons.VoxelToWorld; }
		const DirectX::XMFLOAT4X4& GetWorldToVoxelMatrix() const { return m_cbVoxelCommons.WorldToVoxel; }

		DirectX::GraphicsResource& GetVoxelCommonsResource();

		ConstantBufferVoxelCommons& GetVoxelCommons() { return m_cbVoxelCommons; }

		void BuildMatrices();

		void SetDepthCameraResource(DX12Lib::ConstantBufferLight cameraCB);
		DirectX::GraphicsResource& GetDepthCameraResource();

		void SetLightCameraResource(DX12Lib::ConstantBufferLight cameraCB);
		DirectX::GraphicsResource& GetLightCameraResource();

		void SetDepthCameraSRVHandle(DX12Lib::DescriptorHandle handle) { m_cameraDepthHandleSRV = handle; }
		void SetLightCameraSRVHandle(DX12Lib::DescriptorHandle handle) { m_lightDepthHandleSRV = handle; }

		DX12Lib::DescriptorHandle& GetDepthCameraSRVHandle() { return m_cameraDepthHandleSRV; }
		DX12Lib::DescriptorHandle& GetLightCameraSRVHandle() { return m_lightDepthHandleSRV; }

	private:
		DirectX::XMMATRIX BuildWorldToVoxelMatrix();

	public:
		UINT32 FragmentCount = 0;

		UINT32 MergedClusterCount = 0;
		
		// Number of AABB groups that compose a single geometry
		UINT32 AABBGeometryGroupCount = 0;

		UINT32 FaceCount = 0;

	private:
		UINT32 m_voxelCount;
		UINT32 m_clusterCount;

		std::unique_ptr<TopLevelAccelerationStructure> m_tlas;
		std::unordered_map<std::wstring, std::shared_ptr<BufferManager>> m_bufferManagers;
		DirectX::XMUINT3 VoxelGridSize = DirectX::XMUINT3(128, 128, 128);
		DirectX::XMFLOAT3 VoxelCellSize = DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f);
		DX12Lib::AABB SceneAABB;
		ConstantBufferVoxelCommons m_cbVoxelCommons;
		DirectX::GraphicsResource m_cbVoxelCommonsResource;


		DX12Lib::SceneCamera* m_sceneCamera = nullptr;
		DX12Lib::LightComponent* m_lightComponent = nullptr;

		DirectX::GraphicsResource m_depthCameraResource;
		DirectX::GraphicsResource m_lightCameraResource;

		DX12Lib::DescriptorHandle m_cameraDepthHandleSRV;
		DX12Lib::DescriptorHandle m_lightDepthHandleSRV;
	};


	class Technique
	{
	public:
		Technique() {}
		virtual ~Technique() {}

		virtual void InitializeBuffers() {}
		//virtual void InitializeBuffers(DX12Lib::CommandContext& context, std::shared_ptr<TechniqueData> data) {}
		//virtual void InitializeBuffers(DX12Lib::GraphicsContext& context, std::shared_ptr<TechniqueData> data) {}
		virtual void InitializeBuffers(DX12Lib::ComputeContext& context) {}
		//virtual void InitializeBuffers(RayTracingContext& context, std::shared_ptr<TechniqueData> data) {}


		// Multiple overloads for different contexts in order to avoid dynamic_cast
		virtual void PerformTechnique(DX12Lib::GraphicsContext& context) {}
		virtual void PerformTechnique(DX12Lib::ComputeContext& context) {}
		virtual void PerformTechnique(CVGI::RayTracingContext& context) {}

	protected:
		virtual void TechniquePass(DX12Lib::GraphicsContext& commandContext) {}
		virtual void TechniquePass(DX12Lib::ComputeContext& commandContext, DirectX::XMUINT3 GroupSize) {}
		virtual void TechniquePass(CVGI::RayTracingContext& commandContext, DirectX::XMUINT3 GroupSize) {}

		virtual std::shared_ptr<DX12Lib::RootSignature> BuildRootSignature() { return nullptr; }
		virtual std::shared_ptr<DX12Lib::PipelineState> BuildPipelineState() { return nullptr; }

	protected:
		std::shared_ptr<TechniqueData> m_data = nullptr;
		std::shared_ptr<BufferManager> m_bufferManager = nullptr;

	};
}



