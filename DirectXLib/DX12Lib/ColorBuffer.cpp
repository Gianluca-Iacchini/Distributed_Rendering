#include "ColorBuffer.h"
#include "GraphicsCore.h"

void ColorBuffer::CreateFromSwapChain(ID3D12Resource* baseResource)
{
	AssociateWithResource(baseResource, D3D12_RESOURCE_STATE_PRESENT);

	m_RTVHandle = GraphicsCore::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

	GraphicsCore::s_device->GetComPtr()->CreateRenderTargetView(m_resource.Get(), nullptr, m_RTVHandle);
}
