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


		float zNear = 0.1f;
		float zFar = 100.0f;
		float pad0 = 0;
		float pad1 = 0;
	};

	class VoxelCamera : public DX12Lib::SceneCamera
	{
	public:
		VoxelCamera(DirectX::XMFLOAT3 voxelTexSize = DirectX::XMFLOAT3(128.0f, 128.0f, 128.0f)) :
			DX12Lib::SceneCamera(), m_voxelTexSize(voxelTexSize) {}
		virtual ~VoxelCamera() = default;

		virtual void Init(DX12Lib::CommandContext& context) override;
		virtual DirectX::GraphicsResource GetCameraBuffer() override;

	private:
		DirectX::XMMATRIX BuildViewMatrix(DirectX::XMFLOAT3 axisDirection);

	private:
		ConstantBufferVoxelData m_voxelData;
		DirectX::XMFLOAT3 m_voxelTexSize = DirectX::XMFLOAT3(128.0f, 128.0f, 128.0f);

		// Half sizes of the scene
		DirectX::XMFLOAT3 m_sceneExtents = DirectX::XMFLOAT3(16.0f, 16.0f, 16.0f);
	};

}