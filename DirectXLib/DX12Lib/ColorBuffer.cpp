#include "ColorBuffer.h"
#include "GraphicsCore.h"

using namespace Microsoft::WRL;

void ColorBuffer::CreateFromSwapChain(ComPtr<ID3D12Resource> baseResource)
{
	AssociateWithResource(baseResource, D3D12_RESOURCE_STATE_PRESENT);

	m_RTVHandle = Graphics::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

	Graphics::s_device->GetComPtr()->CreateRenderTargetView(m_resource.Get(), nullptr, m_RTVHandle);
}
