#pragma once

#include "BufferManager.h"

namespace DX12Lib
{
	class ComputeContext;
	class RootSignature;
	class ComputePipelineState;
}

namespace CVGI
{

	class ClusterVisibility
	{
	private:
		__declspec(align(16)) struct ConstantBufferFaceCount
		{
			DirectX::XMUINT3 GridDimension;
			UINT32 CurrentPhase = 0.0f;

			UINT32 VoxelCount = 0;
			DirectX::XMUINT3 pad1;
		};

	public:
		ClusterVisibility(DirectX::XMUINT3 voxelTexDimensions) : m_voxelTexDimensions(voxelTexDimensions) {}
		~ClusterVisibility() {}

		void InitializeBuffers(UINT voxelCount);
		void StartVisibility(BufferManager& compactBufferManager);
		void VisibilityPass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize, BufferManager& compactBufferManager);

		BufferManager* GetBufferManager() { return &m_bufferManager; }
		UINT32 GetFaceCount() { return m_numberOfFaces; }

		std::shared_ptr<DX12Lib::RootSignature> BuildFaceCountRootSignature();
		std::shared_ptr<DX12Lib::ComputePipelineState> BuildFaceCountPipelineState(std::shared_ptr<DX12Lib::RootSignature> rootSig);

	private:
		DirectX::XMUINT3 m_voxelTexDimensions;
		BufferManager m_bufferManager;

		ConstantBufferFaceCount m_cbFaceCount;

		const std::wstring FaceCountPsoName = L"FACE_COUNT_PSO";

		UINT32 m_numberOfVoxels = 0;
		UINT32 m_numberOfFaces = 0;

	private:
		enum class FaceCountRootSignature
		{
			FaceCountCBV = 0,
			CompactSRVTable = 1,
			FaceCountUAVTable,
			Count
		};
	};

}

