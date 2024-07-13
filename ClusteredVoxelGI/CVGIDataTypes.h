#pragma once

#include "DirectXMath.h"
#include "basetsd.h"

namespace CVGI {

	enum class VoxelizeSceneComputeRootParameterSlot
	{
		VoxelCommonCBV = 0,
		ObjectCBV = 1,
		VerticesSRV = 2,
		IndicesSRV = 3,
		MaterialSRV,
		MaterialTextureSRV,
		VoxelTextureUAV,
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
		VoxelTextureUAV = 2,
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


	const D3D12_INPUT_ELEMENT_DESC VertexSingleUINT::InputElements[] =
	{
		{ "SV_Position", 0, DXGI_FORMAT_R32_UINT,    0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
	};

	static_assert(sizeof(VertexSingleUINT) == 4, "Vertex struct/layout mismatch");

	const D3D12_INPUT_LAYOUT_DESC VertexSingleUINT::InputLayout =
	{
		VertexSingleUINT::InputElements,
		VertexSingleUINT::InputElementCount
	};
}