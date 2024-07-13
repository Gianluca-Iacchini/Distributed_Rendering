#pragma once
#include <DX12Lib/Commons/D3DApp.h>
#include <memory>
#include <DirectXMath.h>
#include "DX12Lib/DXWrapper/RootSignature.h"
#include "DX12Lib/DXWrapper/PipelineState.h"
#include "CVGIDataTypes.h"
#include "DX12Lib/DXWrapper/GPUBuffer.h"
#include "DX12Lib/DXWrapper/DescriptorHeap.h"

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
		virtual void Draw(DX12Lib::GraphicsContext& commandContext) override;

		void VoxelPass(DX12Lib::GraphicsContext& context, VoxelCamera* voxelCamera);
		void VoxelDisplayPass(DX12Lib::GraphicsContext& context);
		void VoxelComputePass();

		void CreateSecondVoxelPassBuffers(UINT fragmentCount);
	private:

		std::shared_ptr<DX12Lib::RootSignature> BuildVoxelizeSceneRootSignature();
		std::shared_ptr<DX12Lib::RootSignature> BuildVoxelDisplayRootSignature();
		std::shared_ptr<DX12Lib::RootSignature> BuildVoxelComputeRootSignature();


		std::shared_ptr<DX12Lib::GraphicsPipelineState> BuildVoxelizeScenePso(std::shared_ptr<DX12Lib::RootSignature> voxelRootSig);
		std::shared_ptr<DX12Lib::GraphicsPipelineState> BuildVoxelDisplayPso(std::shared_ptr<DX12Lib::RootSignature> voxelRootSig);
		std::shared_ptr<DX12Lib::ComputePipelineState> BuildVoxelComputePso(std::shared_ptr<DX12Lib::RootSignature> voxelRootSig);

	public:
		const DirectX::XMFLOAT3 VoxelTextureDimension = DirectX::XMFLOAT3(196.0f, 196.0f, 196.0f);

	private:
		ConstantBufferVoxelCommons m_cbVoxelCommons;



		DX12Lib::DescriptorHandle m_voxelDataUAVStart;

		DX12Lib::ByteAddressBuffer m_fragmentCounterBuffer;
		DX12Lib::ByteAddressBuffer m_voxelCounterBuffer;

		// Buffer of size (tex.x * tex.y * tex.z + 31) / 32 keeping track of whether a voxel is occupied or not.
		// It is divided by 32 because it is a buffer of 32 bit uint, where each bit states if the voxel is occupied or not.
		// This *should* be more memory efficient than having a buffer of 1 bool per voxel.
		DX12Lib::StructuredBuffer m_voxelOccupiedBuffer;

		// Buffer of size tex.x * tex.y * tex.z containing the voxel indices of occupied voxel.
		// e.g. if m_voxelIndexBuffer[5] = 2, then voxel with linear coord 5 is occupied by a fragment with index 2 in the
		// fragment data buffer.
		DX12Lib::StructuredBuffer m_voxelIndexBuffer;
		
		// Buffer of size N_Fragments containing the data of all emitted fragments.
		DX12Lib::StructuredBuffer m_fragmentDataBuffer;

		// Buffer that links fragments in the same voxel
		// e.g if m_nextIndexBuffer[5] = 2, then fragment 5 and fragment 2 are in the same voxel.
		DX12Lib::StructuredBuffer m_nextIndexBuffer;

		// Buffer that contains the hashed voxel coordinates of every fragment
		// e.g. if m_hashedBuffer[5] = 2, then fragment 5 is in voxel with hashed coordinate 2.
		// Hashed coordinate is calculated as x + y * tex.x + z * tex.x * tex.y
		DX12Lib::StructuredBuffer m_hashedBuffer;







		DX12Lib::TypedBuffer m_vertexBuffer;

		D3D12_VIEWPORT m_voxelScreenViewport;
		D3D12_RECT m_voxelScissorRect;

		std::vector<DX12Lib::GPUBuffer*> m_uavBuffers;
	};
}