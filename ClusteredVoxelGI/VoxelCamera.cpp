 #include "DX12Lib/pch.h"
#include "VoxelCamera.h"
#include "VoxelScene.h"

using namespace CVGI;

void CVGI::VoxelCamera::Init(DX12Lib::CommandContext& context)
{
	float baseSize = 128.0f;
	DirectX::XMFLOAT3 scaleFactor = {m_voxelTexSize.x / baseSize, m_voxelTexSize.y / baseSize, m_voxelTexSize.z / baseSize};

	this->SetOrthogonal({64, 64, 0.1f, 64});
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
	// Define the scene bounds (example values)
	DirectX::XMFLOAT3 sceneMin = DirectX::XMFLOAT3(-16, -16, -16);
	DirectX::XMFLOAT3 sceneMax = DirectX::XMFLOAT3(16.0f, 16.0f, 16.0f);

	float d = 32.0f;

	// Calculate the scene center
	DirectX::XMFLOAT3 sceneCenter = DirectX::XMFLOAT3(
		(sceneMin.x + sceneMax.x) * 0.5f,
		(sceneMin.y + sceneMax.y) * 0.5f,
		(sceneMin.z + sceneMax.z) * 0.5f
	);


	DirectX::XMFLOAT4X4 view;

	DirectX::XMVECTOR upVector = DirectX::XMVectorSet(0, 1, 0, 0);

	DirectX::XMVECTOR eyePos = DirectX::XMVectorSet(sceneCenter.x - d, sceneCenter.y, sceneCenter.z, 1);


	if (axisDirection.y == 1 || axisDirection.y == -1)
	{
		// If we are looking in the y direction we need to change the up vector
		eyePos = DirectX::XMVectorSet(sceneCenter.x, sceneCenter.y - d, sceneCenter.z, 1);
		upVector = DirectX::XMVectorSet(0, 0, 1, 0);
	}
	else if (axisDirection.z == 1 || axisDirection.z == -1)
	{
		eyePos = DirectX::XMVectorSet(sceneCenter.x, sceneCenter.y, sceneCenter.z - d, 1);
	}

	DirectX::XMVECTOR lookAt = DirectX::XMLoadFloat3(&axisDirection);

	lookAt = DirectX::XMVectorAdd(eyePos, lookAt);

	return DirectX::XMMatrixLookAtLH(eyePos, lookAt, upVector);
}


