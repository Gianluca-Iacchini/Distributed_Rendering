#include "DX12Lib/pch.h"
#include "VoxelCamera.h"
#include "VoxelScene.h"

using namespace CVGI;

void CVGI::VoxelCamera::Init(DX12Lib::CommandContext& context)
{
	this->SetOrthogonal({ 10.0f, 10.0f, 0.1f, 20.f });
}

void VoxelCamera::UseCamera(DX12Lib::CommandContext& context)
{
	if (!IsEnabled) return;

	//DirectX::XMMATRIX xAxisView = GetView();
	DirectX::XMMATRIX xAxisView = this->BuildViewMatrix(DirectX::XMFLOAT3(1, 0, 0));
	DirectX::XMMATRIX yAxisView = this->BuildViewMatrix(DirectX::XMFLOAT3(0, 1, 0));
	DirectX::XMMATRIX zAxisView = this->BuildViewMatrix(DirectX::XMFLOAT3(0, 0, 1));

	DirectX::XMMATRIX proj = this->GetProjection();

	DirectX::XMMATRIX xViewProj = xAxisView * proj;
	DirectX::XMMATRIX yViewProj = yAxisView * proj;
	DirectX::XMMATRIX zViewProj = zAxisView * proj;



	DirectX::XMStoreFloat4x4(&m_voxelData.xAxisView, DirectX::XMMatrixTranspose(xAxisView));
	DirectX::XMStoreFloat4x4(&m_voxelData.yAxisView, DirectX::XMMatrixTranspose(yAxisView));
	DirectX::XMStoreFloat4x4(&m_voxelData.zAxisView, DirectX::XMMatrixTranspose(zAxisView));
	
	DirectX::XMStoreFloat4x4(&m_voxelData.orthoProjection, DirectX::XMMatrixTranspose(proj));
	
	DirectX::XMStoreFloat4x4(&m_voxelData.xAxisViewProjection, DirectX::XMMatrixTranspose(xViewProj));
	DirectX::XMStoreFloat4x4(&m_voxelData.yAxisViewProjection, DirectX::XMMatrixTranspose(yViewProj));
	DirectX::XMStoreFloat4x4(&m_voxelData.zAxisViewProjection, DirectX::XMMatrixTranspose(zViewProj));

	m_voxelData.voxelTextureDimensions = DirectX::XMFLOAT3(128, 128, 128);
	m_voxelData.voxelCellSize = DirectX::XMFLOAT3(1, 1, 1);

	auto cbCamera = Graphics::Renderer::s_graphicsMemory->AllocateConstant(m_voxelData);

	context.m_commandList->Get()->SetGraphicsRootConstantBufferView(
		(UINT)VoxelRootParameter::VoxelDataCBV,
		cbCamera.GpuAddress()
	);
}

DirectX::XMMATRIX VoxelCamera::BuildViewMatrix(DirectX::XMFLOAT3 axisDirection)
{
	DirectX::XMFLOAT4X4 view;

	DirectX::XMVECTOR upVector = DirectX::XMVectorSet(0, 1, 0, 0);

	DirectX::XMVECTOR eyePos = DirectX::XMVectorSet(20, 0, 0, 1);

	if (axisDirection.y == 1 || axisDirection.y == -1)
	{
		// If we are looking in the y direction we need to change the up vector
		upVector = DirectX::XMVectorSet(0, 0, 1, 0);
		eyePos = DirectX::XMVectorSet(0, 20, 0, 1);
	}
	else if (axisDirection.z == 1 || axisDirection.z == -1)
	{
		eyePos = DirectX::XMVectorSet(0, 0, 20, 1);
	}

	return DirectX::XMMatrixLookAtLH(eyePos, DirectX::XMLoadFloat3(&axisDirection), upVector);
}


