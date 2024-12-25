#include "SceneDepthTechnique.h"
#include "DX12Lib/pch.h"
#include "DX12Lib/Scene/SceneCamera.h"
#include "DX12Lib/Scene/LightComponent.h"



using namespace CVGI;
using namespace DX12Lib;
using namespace DirectX;

VOX::SceneDepthTechnique::SceneDepthTechnique(std::shared_ptr<TechniqueData> data, bool cameraOnly) : m_cameraOnly(cameraOnly)
{
	m_data = data;
}

void VOX::SceneDepthTechnique::InitializeBuffers()
{
	auto* sceneCamera = m_data->GetCamera();

	m_lastCameraPosition = sceneCamera->Node->GetPosition();

	assert(sceneCamera != nullptr);

	float aspect = sceneCamera->GetAspect();
	float fovY = sceneCamera->GetFovY();
	float nearZ = sceneCamera->GetNearZ();
	float farZ = sceneCamera->GetFarZ();

	fovY = 2.0f * atan(tan(fovY / 2.0f) * 1.3f);

	m_depthCamera.SetShadowBufferDimensions(1920, 1080);
	m_depthCamera.SetLens(fovY, aspect, nearZ, farZ);
	m_depthCamera.UpdateShadowMatrix(*sceneCamera->Node);

	m_offsetDepthCamera.SetShadowBufferDimensions(1920, 1080);
	m_offsetDepthCamera.SetLens(fovY, aspect, nearZ, farZ);
	m_offsetDepthCamera.UpdateShadowMatrix(*sceneCamera->Node);

	DX12Lib::DescriptorHandle cameraHandle = Graphics::Renderer::s_textureHeap->Alloc(2);

	Graphics::s_device->Get()->CopyDescriptorsSimple(1, cameraHandle, m_depthCamera.GetShadowBuffer().GetDepthSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, cameraHandle + Graphics::Renderer::s_textureHeap->GetDescriptorSize(), m_offsetDepthCamera.GetShadowBuffer().GetDepthSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	m_data->SetDepthCameraSRVHandle(cameraHandle);


	if (m_cameraOnly)
		return;

	auto* lightComp = m_data->GetLightComponent();

	assert(lightComp != nullptr);

	auto* lightShadowCamera = lightComp->GetShadowCamera();

	assert(lightShadowCamera != nullptr);

	m_lightCamera.SetOrthogonalBounds(38.0f, 38.0f, 1.0f, 38.0f);
	m_lightCamera.UpdateShadowMatrix(*lightComp->Node);
	m_lightCamera.SetShadowBufferDimensions(2048, 2048);


	DX12Lib::DescriptorHandle lightHandle = Graphics::Renderer::s_textureHeap->Alloc();

	Graphics::s_device->Get()->CopyDescriptorsSimple(1, lightHandle, m_lightCamera.GetShadowBuffer().GetDepthSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	

	m_data->SetLightCameraSRVHandle(lightHandle);
}



void VOX::SceneDepthTechnique::PerformTechnique(DX12Lib::GraphicsContext& context)
{
	Graphics::Renderer::ShadowPassForCamera(context, &m_depthCamera);
	Graphics::Renderer::ShadowPassForCamera(context, &m_offsetDepthCamera);

	if (!m_cameraOnly)
		Graphics::Renderer::ShadowPassForCamera(context, &m_lightCamera);
}

void VOX::SceneDepthTechnique::UpdateCameraMatrices()
{
	auto* camera = m_data->GetCamera();

	DirectX::XMFLOAT3 cameraPosition = camera->Node->GetPosition();
	DirectX::XMFLOAT3 cameraUp = camera->Node->GetUp();
	DirectX::XMFLOAT3 cameraForward = camera->Node->GetForward();
	DirectX::XMFLOAT3 cameraRight = camera->Node->GetRight();

	DirectX::XMFLOAT3 cameraOffset = DirectX::XMFLOAT3(
		cameraPosition.x - m_lastCameraPosition.x,
		cameraPosition.y - m_lastCameraPosition.y,
		cameraPosition.z - m_lastCameraPosition.z
	);

	DirectX::XMVECTOR offset = DirectX::XMLoadFloat3(&cameraOffset);
	offset = DirectX::XMVector3Normalize(offset);
	offset = DirectX::XMVectorScale(offset, 1.5f);
	offset = DirectX::XMVectorAdd(DirectX::XMLoadFloat3(&cameraPosition), offset);

	DirectX::XMStoreFloat3(&cameraOffset, offset);

	m_lastCameraPosition = cameraPosition;

	m_depthCamera.UpdateShadowMatrix(camera->Node->Transform);
	m_offsetDepthCamera.UpdateShadowMatrix(cameraOffset, cameraUp, cameraForward, cameraRight);

	m_cameraCB.Position = m_data->GetCamera()->Node->GetPosition();
	m_cameraCB.Direction = m_data->GetCamera()->Node->GetForward();
	m_cameraCB.shadowTransform = m_depthCamera.GetShadowTransform();
	m_cameraCB.invShadowTransform = m_depthCamera.GetInvShadowTransform();

	m_offsetCameraCB.Position = cameraOffset;
	m_offsetCameraCB.Direction = m_data->GetCamera()->Node->GetForward();
	m_offsetCameraCB.shadowTransform = m_offsetDepthCamera.GetShadowTransform();
	m_offsetCameraCB.invShadowTransform = m_offsetDepthCamera.GetInvShadowTransform();

	m_data->SetDepthCameraResource(m_cameraCB);
	m_data->SetOffsetDepthCameraResource(m_offsetCameraCB);

	if (!m_cameraOnly)
	{
		m_lightCamera.UpdateShadowMatrix(*m_data->GetLightComponent()->Node);

		m_lightCB.Position = m_data->GetLightComponent()->Node->GetPosition();
		m_lightCB.Direction = m_data->GetLightComponent()->Node->GetForward();
		m_lightCB.shadowTransform = m_lightCamera.GetShadowTransform();
		m_lightCB.invShadowTransform = m_lightCamera.GetInvShadowTransform();

		m_data->SetLightCameraResource(m_lightCB);
	}
}