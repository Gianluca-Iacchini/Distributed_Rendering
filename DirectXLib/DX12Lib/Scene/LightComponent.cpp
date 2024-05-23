#include "DX12Lib/pch.h"
#include "LightComponent.h"


using namespace DX12Lib;
using namespace Graphics;

std::vector<LightComponent*> LightComponent::m_activeLights;
DirectX::GraphicsResource LightComponent::m_lightBufferSRV;

void DX12Lib::LightComponent::UpdateLights(CommandContext& context)
{

	// Erase the nullptr elements from the vector
	auto newEnd = std::remove_if(m_activeLights.begin(), m_activeLights.end(),
		[](const LightComponent* ptr) { return ptr == nullptr; });
	m_activeLights.erase(newEnd, m_activeLights.end());

	m_lightBufferSRV = Renderer::s_graphicsMemory->Allocate(sizeof(ConstantBufferLight) * m_activeLights.size());
}

void DX12Lib::LightComponent::RenderLights(CommandContext& context)
{
	context.m_commandList->Get()->SetGraphicsRootShaderResourceView(
		(UINT)Renderer::RootSignatureSlot::LightSRV, m_lightBufferSRV.GpuAddress());
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
	DXLIB_CORE_INFO("LightComponent destroyed");
	RemoveLight(m_lightIndex);
}

void DX12Lib::LightComponent::Update(CommandContext& context)
{
	m_lightCB.Direction = this->Node->GetForward();
	m_lightCB.Position = this->Node->GetPosition();
	
	this->Node->Rotate(Node->GetRight(), this->Node->Scene.Time().DeltaTime());

	memcpy(m_lightBufferSRV.Memory(), &m_lightCB, sizeof(m_lightCB));
}

void DX12Lib::LightComponent::Render(CommandContext& context)
{

}


