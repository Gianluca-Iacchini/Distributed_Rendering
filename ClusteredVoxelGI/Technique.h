#pragma once

#include <memory>
#include "BufferManager.h"
#include "Shaders/TechniquesCompat.h"
#include <unordered_map>
#include "RayTracingHelpers.h"
#include "WinPixEventRuntime/pix3.h"

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

		BufferManager& GetBufferManager(std::wstring name)
		{
			auto it = m_bufferManagers.find(name);
			assert(it != m_bufferManagers.end() && "Buffer Manager not found");
			return *(it->second);
		}

		void AddBufferManager(std::wstring name, std::shared_ptr<BufferManager> bufferManager)
		{
			m_bufferManagers[name] = bufferManager;
		}

		void SetTlas(std::unique_ptr<TopLevelAccelerationStructure>&& tlas) { m_tlas = std::move(tlas); }
		const TopLevelAccelerationStructure* GetTlas() { return m_tlas.get(); }

	public:
		DirectX::XMUINT3 VoxelGridSize = DirectX::XMUINT3(128, 128, 128);
		UINT32 VoxelCount = 0;
		UINT32 FragmentCount = 0;
		UINT32 ClusterCount = 0;
		UINT32 MergedClusterCount = 0;
		UINT32 FaceCount = 0;

	private:
		std::unique_ptr<TopLevelAccelerationStructure> m_tlas;
		std::unordered_map<std::wstring, std::shared_ptr<BufferManager>> m_bufferManagers;
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



