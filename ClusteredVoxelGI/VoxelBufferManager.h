#pragma once

#include "DX12Lib/DXWrapper/GPUBuffer.h"
#include "DX12Lib/DXWrapper/DescriptorHeap.h"
#include "DX12Lib/DXWrapper/ColorBuffer.h"
#include "DirectXMath.h"
#include "CVGIDataTypes.h"

namespace DX12Lib
{
	class RootSignature;
	class ComputePipelineState;
	class ComputeContext;
}

namespace Graphics {
	namespace Renderer {
		extern std::shared_ptr<DX12Lib::DescriptorHeap> s_textureHeap;
	}
}
namespace CVGI
{

	enum class BufferType
	{
		// Voxelization
		FragmentData = 0,
		NextIndex = 1,
		VoxelIndex = 2,
		FragmentCounter = 3,
		VoxelCounter,
		VoxelOccupied,
		HashedBuffer,

		// Compaction
		IndirectionRankBuffer,
		IndirectionIndexBuffer,
		CompactedVoxelIndex,
		CompactedHashedBuffer,
		PrefixSum,

		// Clusterization
		ClusterData,
		NextVoxel,
		AssignmentMap,
		DistanceMap,
		TileBuffer,
		NextCluster,
		ClusterCounterBuffer,

		Count
	};

	class VoxelBufferManager
	{
	public:
		VoxelBufferManager();
		~VoxelBufferManager() = default;

		void SetupFirstVoxelPassBuffers(DirectX::XMFLOAT3 voxelTexDimension);
		void SetupSecondVoxelPassBuffers(DX12Lib::CommandContext& context, UINT numFragments);

		void SetVoxelTextureDimension(DirectX::XMFLOAT3 voxelTexDimension) 
		{ 
			m_voxelTexDimension = voxelTexDimension;
			m_voxelLinearSize = (UINT)(m_voxelTexDimension.x * m_voxelTexDimension.y * m_voxelTexDimension.z);
		}

		inline DX12Lib::GPUBuffer* GetBuffer(BufferType type) { return m_buffers[(UINT)type]; }
		DX12Lib::DescriptorHandle& GetBufferUAVStart(BufferType type);
		DX12Lib::DescriptorHandle& GetBufferSRVStart(BufferType type);

		inline DX12Lib::DescriptorHandle& GetVoxelizeTableUAV() { return m_voxelizationUAVStart; }
		inline DX12Lib::DescriptorHandle& GetCompactionTableUAV() { return m_streamCompactUAVStart; }
		inline DX12Lib::DescriptorHandle& GetClusterizeTableUAV() { return m_clusterizeUAVStart; }

		inline DX12Lib::DescriptorHandle& GetVoxelizeTableSRV() { return m_voxelizationSRVStart; }
		inline DX12Lib::DescriptorHandle& GetCompactionTableSRV() { return m_streamCompactSRVStart; }
		inline DX12Lib::DescriptorHandle& GetClusterizeTableSRV() { return m_clusterizeSRVStart; }

		template <typename T>
		T ReadFromBuffer(DX12Lib::CommandContext& context, BufferType type);

		void SetupCompactBuffers();

		void CompactBuffers();
		void ClusterizeBuffers();

		void InitializeClusters();

		std::shared_ptr<DX12Lib::RootSignature> BuildCompactBufferRootSignature();
		std::shared_ptr<DX12Lib::ComputePipelineState> BuildCompactBufferPso(std::shared_ptr<DX12Lib::RootSignature> voxelRootSig);

		std::shared_ptr<DX12Lib::RootSignature> BuildClusterizeRootSignature();
		std::shared_ptr<DX12Lib::ComputePipelineState> BuildClulsterizePso(std::shared_ptr<DX12Lib::RootSignature> voxelRootSig);

		UINT32 GetNumberOfVoxels() { return m_voxelCount; }

	private:
		void CompactBufferPass(DX12Lib::ComputeContext& context, UINT numGroupsX);
		void ClusterizeBufferPass(DX12Lib::ComputeContext& context);

	public:
		static const std::wstring CompactBufferPsoName;
		static const std::wstring ClusterizeBufferPsoName;

	private:
#pragma region CompactVariables
		ConstantBufferCompactBuffer m_cbCompactBuffer;
		ConstantBufferClusterizeBuffer m_cbClusterizeBuffer;

		const UINT m_elementsPerThread = 128;

		DirectX::XMFLOAT3 m_voxelTexDimension = DirectX::XMFLOAT3(128, 128, 128);
		UINT m_voxelLinearSize = 0;

		UINT m_fragmentCount = 0;
		UINT m_voxelCount = 0;

		UINT m_reduceStepCount = 0;
		UINT m_downSweepStepCount = 0;

		UINT m_currentStep = 0;
		UINT m_currentPhase = 0;

		// Size of the prefix sum buffer for the current step
		std::vector<UINT32> v_prefixBufferSizeForStep;
		UINT64 m_prefixBufferSize = 0;
		bool m_firstSetIsSingleElement = false;
#pragma endregion

#pragma region ClusterVariables
		UINT32 m_numberOfClusters = 1;
		UINT32 m_compactness = 10;
		float m_superPixelArea = 1;

		DirectX::XMUINT3 TileGridDimension;

#pragma endregion

#pragma region Buffers
		//DX12Lib::DescriptorHandle m_voxelDataUAVStart;
		DX12Lib::DescriptorHandle m_voxelizationUAVStart;
		DX12Lib::DescriptorHandle m_streamCompactUAVStart;
		DX12Lib::DescriptorHandle m_clusterizeUAVStart;

		DX12Lib::DescriptorHandle m_voxelizationSRVStart;
		DX12Lib::DescriptorHandle m_streamCompactSRVStart;
		DX12Lib::DescriptorHandle m_clusterizeSRVStart;

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

		// Buffer to perfrom prefix sum on the voxelIndexBuffer
		DX12Lib::StructuredBuffer m_prefixSumBuffer;

		DX12Lib::StructuredBuffer m_compactedVoxelIndexBuffer;
		DX12Lib::StructuredBuffer m_compactedHashedBuffer;

		DX12Lib::StructuredBuffer m_indirectionRankBuffer;
		DX12Lib::StructuredBuffer m_indirectionIndexBuffer;

		// Clusterization
		DX12Lib::StructuredBuffer m_clusterDataBuffer;
		DX12Lib::StructuredBuffer m_assignemtMapBuffer;
		DX12Lib::StructuredBuffer m_voxelClusterLinkedList;
		DX12Lib::StructuredBuffer m_distanceMapBuffer;
		DX12Lib::StructuredBuffer m_nextClusterList;
		DX12Lib::StructuredBuffer m_clusterCounterBuffer;
		DX12Lib::ColorBuffer m_tileTexture;


		DX12Lib::GPUBuffer* m_buffers[(UINT)BufferType::Count];

#pragma endregion
	};

	template<typename T>
	inline T VoxelBufferManager::ReadFromBuffer(DX12Lib::CommandContext& context, BufferType type)
	{
		GPUBuffer* buffer = m_buffers[(UINT)type];

		ReadBackBuffer readBuffer;
		readBuffer.Create(buffer->GetElementCount(), buffer->GetElementSize());



		context.CopyBuffer(readBuffer, *buffer);

		context.Flush(true);

		void* data =  readBuffer.ReadBack(*buffer);

		if (type == BufferType::FragmentCounter)
			m_fragmentCount = *reinterpret_cast<UINT*>(data);
		else if (type == BufferType::VoxelCounter)
			m_voxelCount = *reinterpret_cast<UINT*>(data);

		return reinterpret_cast<T>(data);
	}

}


