#pragma once

#include "DirectXMath.h"
#include "basetsd.h"
#include "directx/d3dx12.h"

namespace VOX {

	enum class CompactBufferRootSignature
	{
		PrefixSumCBV = 0,
		VoxelizeUAVTable = 1,
		StreamCompactionUAVTable = 2,
		Count
	};

	enum class ClusterizeRootSignature
	{
		ClusterizeCBV = 0,
		VoxelBuffersSRVTable = 1,
		StreamCompactionSRVTable = 2,
		ClusterizeUAVTable,
		Count
	};

	enum class ClusterReduceRootSignature
	{
		ClusterReduceCBV = 0,
		ClusterizeSRVTable = 1,
		ClusterizeUAVTable = 2,
		Count
	};

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

	__declspec(align(16)) struct ConstantBufferCompactBuffer
	{
		UINT32 CurrentPhase = 0;
		UINT32 CurrentStep = 0;
		UINT32 CompactBufferSize = 0;
		UINT32 ElementsPerThread = 128;

		UINT32 NumElementsSweepDown = 0;
		UINT32 NumElementsBase;
		UINT32 NumElementsLevel0;
		UINT32 NumElementsLevel1;

		UINT32 NumElementsLevel2;
		UINT32 NumElementsLevel3;
		float pad0 = 0.0f;
		float pad1 = 0.0f;

		DirectX::XMUINT3 VoxelTextureDimensions = DirectX::XMUINT3(128, 128, 128);
		float pad2 = 0.0f;
	};

	__declspec(align(16)) struct ConstantBufferClusterizeBuffer
	{
		UINT32 CurrentPhase = 0;
		UINT32 VoxelCount = 0;
		UINT32 K = 10000;
		float m = 1.0f;

		DirectX::XMUINT3 VoxelTextureDimensions = DirectX::XMUINT3(128, 128, 128);
		UINT32 S = 1;

		DirectX::XMUINT3 TileGridDimension = DirectX::XMUINT3(6, 6, 6);
		UINT32 FirstClusterSet = 1;

		DirectX::XMUINT3 CurrentTileUpdate = DirectX::XMUINT3(0, 0, 0);
		float pad1 = 0.0f;
	}; 

	__declspec(align(16)) struct ConstantBufferClusterReduce
	{
		UINT32 CurrentStep = 0;
		UINT32 NumberOfSubClusters = 0;
		UINT32 NumberOfSuperClusters = 0;
		float Compactness = 10.0f;

		DirectX::XMUINT3 VoxelDimension = DirectX::XMUINT3(512, 512, 512);
		UINT32 S = 1;

		DirectX::XMUINT3 TileGridDimension = DirectX::XMUINT3(6, 6, 6);
		UINT32 FirstClusterSet = UINT_MAX;

		UINT32 CurrentIteration = 1;
		UINT32 VoxelCount = 0;
		float _pad1;
		float _pad2;
	};


	__declspec(align(16)) struct FragmentData
	{
		DirectX::XMFLOAT3 Position;
		float pad0;

		DirectX::XMFLOAT4 Color;

		DirectX::XMFLOAT3 Normal;
		unsigned int VoxelLinearCoord;
	};

	__declspec(align(16)) struct ClusterData
	{
		DirectX::XMFLOAT3 Center;
		UINT32 VoxelCount;

		DirectX::XMFLOAT3 Normal;
		UINT32 FirstVoxelDataIndex;

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

	
}