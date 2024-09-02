#pragma once

#include <DirectXMath.h>
#include "BufferManager.h"
#include "DX12Lib/Commons/CommandContext.h"
#include "VoxelCamera.h"

namespace DX12Lib
{
	class RootSignature;
	class GraphicsPipelineState;
}

namespace CVGI
{

	class VoxelizeScene
	{
	private:
		enum class VoxelizeSceneRootParameterSlot
		{
			VoxelCommonCBV = 0,
			VoxelCameraCBV = 1,
			ObjectCBV = 2,
			MaterialSRV,
			MaterialTextureSRV,
			VoxelDataUAV,
			Count
		};

		enum class DisplayVoxelRootParameterSlot
		{
			VoxelCommonCBV = 0,
			CameraCBV = 1,
			VoxelSRVBufferTable = 2,
			CompactSRVBufferTable,
			ClusterSRVBufferTable,
			ClusterUAVBufferTable,
			Count
		};

		__declspec(align(16)) struct ConstantBufferVoxelCommons
		{

			DirectX::XMUINT3 voxelTextureDimensions = DirectX::XMUINT3(128, 128, 128);
			float totalTime = 0.0f;

			DirectX::XMFLOAT3 voxelCellSize = DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f);
			float deltaTime = 0.0f;

			DirectX::XMFLOAT3 invVoxelTextureDimensions = DirectX::XMFLOAT3(1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f);
			UINT32 StoreData = 0;

			DirectX::XMFLOAT3 invVoxelCellSize = DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f);
			float pad1 = 0.0f;

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
			FragmentData = 0,
			NextIndex = 1,
			VoxelIndex = 2,
			FragmentCounter = 3,
			VoxelCounter,
			VoxelOccupied,
			HashedBuffer,
		};

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

	public:
		VoxelizeScene(DirectX::XMUINT3 VoxelSceneSize, DirectX::XMFLOAT3 VoxelSize);
		~VoxelizeScene() {}

		void InitializeBuffers();
		void UpdateBuffers(DX12Lib::CommandContext& context);
		
		void SetVertexData(DX12Lib::CommandContext& context, UINT32 vertexCount);
		
		void VoxelizePass(DX12Lib::GraphicsContext& context, VoxelCamera* voxelCamera);
		void DisplayVoxelPass(DX12Lib::GraphicsContext& context, DX12Lib::SceneCamera* camera, BufferManager* compactBufferManager);

		DX12Lib::GPUBuffer& GetVoxelBuffer(VoxelBufferType type) { return m_bufferManager.GetBuffer((UINT)type); }

		BufferManager* GetBufferManager() { return &m_bufferManager; }

		std::shared_ptr<DX12Lib::RootSignature> BuildVoxelizeSceneRootSignature();
		std::shared_ptr<DX12Lib::GraphicsPipelineState> BuildVoxelizeScenePso(std::shared_ptr<DX12Lib::RootSignature> voxelRootSig);

		std::shared_ptr<DX12Lib::RootSignature> BuildDisplayVoxelRootSignature();
		std::shared_ptr<DX12Lib::GraphicsPipelineState> BuildDisplayVoxelPso(std::shared_ptr<DX12Lib::RootSignature> voxelRootSig);

	private:
		DirectX::XMUINT3 m_voxelSceneDimensions = DirectX::XMUINT3(128, 128, 128);

		DX12Lib::TypedBuffer m_vertexBuffer;
		UINT64 m_vertexCount = 0;

		ConstantBufferVoxelCommons m_cbVoxelCommons;
		BufferManager m_bufferManager;

		D3D12_VIEWPORT m_voxelScreenViewport;
		D3D12_RECT m_voxelScissorRect;

		UINT32 m_fragmentCount = 0;
		UINT32 m_voxelCount = 0;
	};
}


