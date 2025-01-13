#include "DX12Lib/pch.h"
#include "LightComponent.h"
#include "DX12Lib/Commons/ShadowMap.h"

using namespace DX12Lib;
using namespace Graphics;

std::vector<LightComponent*> LightComponent::m_activeLights;
DirectX::GraphicsResource LightComponent::m_lightBufferSRV;
ConstantBufferLight* LightComponent::s_lightBufferData = nullptr;

void DX12Lib::LightComponent::SetCastsShadows(bool value)
{
	// For now only shadow for directional lights are supported
	if (m_lightType != LightType::Directional)
	{
		m_doesCastShadows = false;
		return;
	}

	if (value)
	{
		if (m_shadowCamera == nullptr)
		{
			m_shadowCamera = std::make_unique<ShadowCamera>();
			m_shadowCamera->SetOrthogonalBounds(38.0f, 38.0f, 1.0f, 38.0f);
			m_shadowCamera->UpdateShadowMatrix(*this->Node);
		}

		// Move the light to the front of the active lights vector
		std::iter_swap(m_activeLights.begin(), m_activeLights.begin() + m_lightIndex);
		m_activeLights[m_lightIndex]->m_lightIndex = m_lightIndex;
		m_lightIndex = 0;
	}

	m_doesCastShadows = value;
}

ShadowCamera* DX12Lib::LightComponent::GetShadowCamera()
{
	assert(m_shadowCamera != nullptr);
	return m_shadowCamera.get();
}

void DX12Lib::LightComponent::UpdateLights(CommandContext& context)
{

	// Erase the nullptr elements from the vector
	auto newEnd = std::remove_if(m_activeLights.begin(), m_activeLights.end(),
		[](const LightComponent* ptr) { return ptr == nullptr; });
	m_activeLights.erase(newEnd, m_activeLights.end());

	if (m_activeLights.size() > 0)
	{
		m_lightBufferSRV = Renderer::s_graphicsMemory->Allocate(sizeof(ConstantBufferLight) * m_activeLights.size());
	}
	else
	{
		// If there are no lights create a dummy light with no intensity
		m_lightBufferSRV = Renderer::s_graphicsMemory->Allocate(sizeof(ConstantBufferLight));
		memset(m_lightBufferSRV.Memory(), 0, sizeof(m_lightCB));
	}

	s_lightBufferData = reinterpret_cast<ConstantBufferLight*>(m_lightBufferSRV.Memory());
}

void DX12Lib::LightComponent::RenderLights(CommandContext& context)
{
	context.m_commandList->Get()->SetGraphicsRootShaderResourceView(
		(UINT)Renderer::RootSignatureSlot::LightSRV, m_lightBufferSRV.GpuAddress());

	for (auto light : m_activeLights)
		if (light->CastsShadows())
			Graphics::Renderer::AddShadowCamera(light->GetShadowCamera());
}

void LightComponent::RemoveLight(int index)
{
	if (index < 0 || index >= m_activeLights.size())
		return;

	m_activeLights.erase(m_activeLights.begin() + index);
	for (int i = index; i < m_activeLights.size(); i++)
	{
		m_activeLights[i]->m_lightIndex = i;
	}
}

DX12Lib::LightComponent::LightComponent()
{
	m_lightIndex = m_activeLights.size();
	m_activeLights.push_back(this);
}

DX12Lib::LightComponent::~LightComponent()
{
	RemoveLight(m_lightIndex);
}

void DX12Lib::LightComponent::Init(CommandContext& context)
{
	m_shadowCamera->UpdateShadowMatrix(*this->Node);
}

void DX12Lib::LightComponent::Update(CommandContext& context)
{

	m_lightCB.Direction = this->Node->GetForward();
	m_lightCB.Position = this->Node->GetPosition();
	m_lightCB.CastsShadows = (int)this->m_doesCastShadows;
	m_lightCB.Color.x = m_color.x * m_intensity;
	m_lightCB.Color.y = m_color.y * m_intensity;
	m_lightCB.Color.z = m_color.z * m_intensity;

	memcpy(s_lightBufferData + m_lightIndex, &m_lightCB, sizeof(m_lightCB));

	if (this->m_doesCastShadows)
	{
		if (Node->IsTransformDirty() && m_shadowCamera != nullptr)
		{
			m_shadowCamera->UpdateShadowMatrix(*this->Node);
			m_lightCB.shadowTransform = m_shadowCamera->GetShadowTransform();
			m_lightCB.invShadowTransform = m_shadowCamera->GetInvShadowTransform();
		}
	}

	m_didLightPropertyChange = false;
}

void DX12Lib::LightComponent::Render(CommandContext& context)
{

}

void DX12Lib::LightComponent::SetLightColor(DirectX::XMFLOAT3 color)
{
	m_color = color;
	m_didLightPropertyChange = true;
}

void DX12Lib::LightComponent::SetLightIntensity(float intensity)
{
	m_intensity = intensity;
	m_didLightPropertyChange = true;
}


