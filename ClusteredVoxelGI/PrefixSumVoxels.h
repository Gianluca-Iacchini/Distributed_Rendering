#pragma once
#include "BufferManager.h"
#include "DirectXMath.h"

namespace DX12Lib
{
	class CommandContext;
	class ComputeContext;
	class RootSignature;
	class ComputePipelineState;
}

namespace CVGI
{
	class PrefixSumVoxels
	{
	private:
		__declspec(align(16)) struct ConstantBufferCompactBuffer
		{
			UINT32 CurrentPhase = 0;
			UINT32 CurrentStep = 0;
			UINT32 CompactBufferSize = 0;
			UINT32 ElementsPerThread = 128;

			UINT32 NumElementsSweepDown = 0;
			UINT32 NumElementsBase = 0;
			UINT32 NumElementsLevel0 = 0;
			UINT32 NumElementsLevel1 = 0;

			UINT32 NumElementsLevel2 = 0;
			UINT32 NumElementsLevel3 = 0;
			float pad0 = 0.0f;
			float pad1 = 0.0f;

			DirectX::XMUINT3 VoxelTextureDimensions = DirectX::XMUINT3(128, 128, 128);
			float pad2 = 0.0f;
		};

		enum class PrefixSumBufferType
		{
			IndirectionRankBuffer,
			IndirectionIndexBuffer,
			CompactedVoxelIndex,
			CompactedHashedBuffer,
			PrefixSum,
		};

		enum class CompactBufferRootSignature
		{
			PrefixSumCBV = 0,
			VoxelizeUAVTable = 1,
			StreamCompactionUAVTable = 2,
			Count
		};

	public:
		PrefixSumVoxels(DirectX::XMUINT3 VoxelizationSize) : m_voxelizationSize(VoxelizationSize) {}
		~PrefixSumVoxels() {}

		BufferManager* GetBufferManager() { return &m_bufferManager; }

		void InitializeBuffers(DX12Lib::CommandContext& context);

		void StartPrefixSum(BufferManager* voxelBufferManager);
		void CompactBufferPass(DX12Lib::ComputeContext& context, UINT32 numGroupsX, BufferManager* voxelBufferManager);

		std::shared_ptr<DX12Lib::RootSignature> BuildPrefixSumRootSignature();
		std::shared_ptr<DX12Lib::ComputePipelineState> BuildPrefixSumPipelineState(std::shared_ptr<DX12Lib::RootSignature> rootSig);

	private:
		void ComputePrefixSumVariables();

	private:

		BufferManager m_bufferManager;

		ConstantBufferCompactBuffer m_cbCompactBuffer;

		DirectX::XMUINT3 m_voxelizationSize = DirectX::XMUINT3(128, 128, 128);

		// Size of the prefix sum buffer for the current step
		std::vector<UINT32> v_prefixBufferSizeForStep;
		UINT64 m_prefixBufferSize = 0;
		bool m_firstSetIsSingleElement = false;

		UINT m_reduceStepCount = 0;
		UINT m_downSweepStepCount = 0;

		UINT m_currentStep = 0;
		UINT m_currentPhase = 0;

		const UINT ELEMENTS_PER_THREAD = 128;
		const std::wstring PrefixSumPsoName = L"PREFIX_SUM_PSO";
	};
}

