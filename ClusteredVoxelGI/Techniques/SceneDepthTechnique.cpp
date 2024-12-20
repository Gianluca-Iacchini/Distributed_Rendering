#include "SceneDepthTechnique.h"
#include "DX12Lib/pch.h"
#include "DX12Lib/Scene/SceneCamera.h"
#include "DX12Lib/Scene/LightComponent.h"



using namespace CVGI;
using namespace DX12Lib;
using namespace DirectX;

CVGI::SceneDepthTechnique::SceneDepthTechnique(std::shared_ptr<TechniqueData> data)
{
	m_data = data;
}

void CVGI::SceneDepthTechnique::InitializeBuffers()
{
	auto* sceneCamera = m_data->GetCamera();

	assert(sceneCamera != nullptr);

	float aspect = sceneCamera->GetAspect();
	float fovY = sceneCamera->GetFovY();
	float nearZ = sceneCamera->GetNearZ();
	float farZ = sceneCamera->GetFarZ();

	fovY = 2.0f * atan(tan(fovY / 2.0f) * 1.3f);

	m_depthCamera.SetShadowBufferDimensions(1920, 1080);
	m_depthCamera.SetLens(fovY, aspect, nearZ, farZ);
	m_lightCamera.UpdateShadowMatrix(*sceneCamera->Node);

	auto* lightComp = m_data->GetLightComponent();

	assert(lightComp != nullptr);

	auto* lightShadowCamera = lightComp->GetShadowCamera();

	assert(lightShadowCamera != nullptr);

	m_lightCamera.SetOrthogonalBounds(38.0f, 38.0f, 1.0f, 38.0f);
	m_lightCamera.UpdateShadowMatrix(*lightComp->Node);
	m_lightCamera.SetShadowBufferDimensions(2048, 2048);

	DX12Lib::DescriptorHandle cameraHandle = Graphics::Renderer::s_textureHeap->Alloc();
	DX12Lib::DescriptorHandle lightHandle = Graphics::Renderer::s_textureHeap->Alloc();

	Graphics::s_device->Get()->CopyDescriptorsSimple(1, cameraHandle, m_depthCamera.GetShadowBuffer().GetDepthSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	Graphics::s_device->Get()->CopyDescriptorsSimple(1, lightHandle, m_lightCamera.GetShadowBuffer().GetDepthSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	

	m_data->SetDepthCameraSRVHandle(cameraHandle);
	m_data->SetLightCameraSRVHandle(lightHandle);
}



void CVGI::SceneDepthTechnique::PerformTechnique(DX12Lib::GraphicsContext& context)
{
	Graphics::Renderer::ShadowPassForCamera(context, &m_depthCamera);
	Graphics::Renderer::ShadowPassForCamera(context, &m_lightCamera);
}

void CVGI::SceneDepthTechnique::UpdateCameraMatrices()
{
	m_depthCamera.UpdateShadowMatrix(*m_data->GetCamera()->Node);
	m_lightCamera.UpdateShadowMatrix(*m_data->GetLightComponent()->Node);

	m_cameraCB.Position = m_data->GetCamera()->Node->GetPosition();
	m_cameraCB.Direction = m_data->GetCamera()->Node->GetForward();
	m_cameraCB.shadowTransform = m_depthCamera.GetShadowTransform();
	m_cameraCB.invShadowTransform = m_depthCamera.GetInvShadowTransform();

	m_lightCB.Position = m_data->GetLightComponent()->Node->GetPosition();
	m_lightCB.Direction = m_data->GetLightComponent()->Node->GetForward();
	m_lightCB.shadowTransform = m_lightCamera.GetShadowTransform();
	m_lightCB.invShadowTransform = m_lightCamera.GetInvShadowTransform();

	m_data->SetDepthCameraResource(m_cameraCB);
	m_data->SetLightCameraResource(m_lightCB);

}
