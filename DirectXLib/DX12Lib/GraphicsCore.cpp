#include "GraphicsCore.h"
#include "Device.h"

void GraphicsCore::Initialize(Device* device)
{
	s_descriptorAllocators = new DescriptorAllocator[D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES]
	{
		DescriptorAllocator(device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
		DescriptorAllocator(device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER),
		DescriptorAllocator(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV),
		DescriptorAllocator(device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV)
	};

	s_device = device;
}

DescriptorAllocator* GraphicsCore::s_descriptorAllocators = nullptr;
Device* GraphicsCore::s_device = nullptr;
