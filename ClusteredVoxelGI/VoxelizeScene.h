#pragma once

#include <DirectXMath.h>
#include "Technique.h"
#include "VoxelCamera.h"


namespace CVGI
{

	class VoxelizeScene : public Technique
	{
	private:
		enum class VoxelizeSceneRootParameterSlot
		{
			VoxelCommonCBV = 0,
			VoxelCameraCBV = 1,
			ObjectCBV = 2,
			VoxelConstantCBV,
			MaterialSRV,
			MaterialTextureSRV,
			VoxelDataUAV,
			Count
		};

		__declspec(align(16)) struct FragmentData
		{
			DirectX::XMFLOAT3 Position;
			float pad0;

			DirectX::XMFLOAT4 Color;

			DirectX::XMFLOAT3 Normal;
			unsigned int VoxelLinearCoord;
		};

	public:
		enum class VoxelBufferType
		{
			VoxelOccupied = 0,
			FragmentData = 1,
			NextIndex = 2,
			VoxelIndex,
			FragmentCounter,
			VoxelCounter,
			HashedBuffer,
		};



	public:
		VoxelizeScene(std::shared_ptr<TechniqueData> data)
		{
			m_bufferManager = std::make_shared<BufferManager>();
			data->SetBufferManager(Name, m_bufferManager);
			m_data = data;
		}
		virtual ~VoxelizeScene() {}

		virtual void InitializeBuffers() override;
		void UpdateBuffers(DX12Lib::CommandContext& context);

		void SetVoxelCamera(VoxelCamera* camera);
		virtual void PerformTechnique(DX12Lib::GraphicsContext& context) override;
		virtual void TechniquePass(DX12Lib::GraphicsContext& context) override;


		void DeleteTemporaryBuffers();

		virtual std::shared_ptr<DX12Lib::RootSignature> BuildRootSignature() override;
		virtual std::shared_ptr<DX12Lib::PipelineState> BuildPipelineState() override;


	public:
		static const std::wstring Name;

	private:
		VoxelCamera* m_voxelCamera = nullptr;

		UINT32 m_currentPass = 0;

		D3D12_VIEWPORT m_voxelScreenViewport;
		D3D12_RECT m_voxelScissorRect;

	};
}


