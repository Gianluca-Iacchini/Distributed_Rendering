 #include "DX12Lib/pch.h"
#include "VoxelCamera.h"
#include "VoxelScene.h"

using namespace CVGI;

void CVGI::VoxelCamera::Init(DX12Lib::CommandContext& context)
{
	// SetOrthogonal takes the bounding box width, height, near value and far values, which equals to 2 * half the size of the scene
	this->SetOrthogonal({ m_sceneHalfExtents.x * 2.f, m_sceneHalfExtents.y * 2.f, 0.1f, m_sceneHalfExtents.z * 2.f});
}

DirectX::GraphicsResource VoxelCamera::GetCameraBuffer()
{
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

	m_voxelData.zNear = this->GetNearZ();
	m_voxelData.zFar = this->GetFarZ();

	return Graphics::Renderer::s_graphicsMemory->AllocateConstant(m_voxelData);
}

DirectX::XMMATRIX VoxelCamera::BuildViewMatrix(DirectX::XMFLOAT3 axisDirection)
{
	// In our case the scene is centered at the origin, so the camera is offset at a distance equal to the scene's half size
	DirectX::XMFLOAT3 cameraDistance = m_sceneHalfExtents;

	DirectX::XMFLOAT4X4 view;

	DirectX::XMVECTOR upVector = DirectX::XMVectorSet(0, 1, 0, 0);

	DirectX::XMVECTOR eyePos = DirectX::XMVectorSet(-cameraDistance.x, 0, 0, 1);


	if (axisDirection.y == 1 || axisDirection.y == -1)
	{
		// If we are looking in the y direction we need to change the up vector
		eyePos = DirectX::XMVectorSet(0, -cameraDistance.y, 0, 1);
		upVector = DirectX::XMVectorSet(0, 0, 1, 0);
	}
	else if (axisDirection.z == 1 || axisDirection.z == -1)
	{
		eyePos = DirectX::XMVectorSet(0, 0, -cameraDistance.z, 1);
	}

	DirectX::XMVECTOR lookAt = DirectX::XMLoadFloat3(&axisDirection);

	lookAt = DirectX::XMVectorAdd(eyePos, lookAt);

	return DirectX::XMMatrixLookAtLH(eyePos, lookAt, upVector);
}


