#pragma once

#include "DX12Lib/Scene/SceneCamera.h"

namespace CVGI
{
	__declspec(align(16)) struct ConstantBufferVoxelData
	{
		DirectX::XMFLOAT4X4 xAxisView = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 yAxisView = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 zAxisView = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 orthoProjection = MathHelper::Identity4x4();

		DirectX::XMFLOAT4X4 xAxisViewProjection = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 yAxisViewProjection = MathHelper::Identity4x4();
		DirectX::XMFLOAT4X4 zAxisViewProjection = MathHelper::Identity4x4();

		DirectX::XMFLOAT3 voxelTextureDimensions = DirectX::XMFLOAT3(128.0f, 128.0f, 128.0f);

		DirectX::XMFLOAT3 voxelCellSize = DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f);
	};

	class VoxelCamera : public DX12Lib::SceneCamera
	{
	public:
		VoxelCamera() : DX12Lib::SceneCamera() {}
		virtual ~VoxelCamera() = default;

		void Init(DX12Lib::CommandContext& context) override;
		void UseCamera(DX12Lib::CommandContext& context) override;

	private:
		DirectX::XMMATRIX BuildViewMatrix(DirectX::XMFLOAT3 axisDirection);

	private:
		ConstantBufferVoxelData m_voxelData;
	};

}