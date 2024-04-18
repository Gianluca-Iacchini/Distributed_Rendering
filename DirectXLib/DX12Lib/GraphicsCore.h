#ifndef  GRAPHICS_CORE_H
#define GRAPHICS_CORE_H
#include "DX12Lib/DescriptorHeap.h"

class Device;

class GraphicsCore
{
public:
	static void Initialize(Device* device);
	static D3D12_CPU_DESCRIPTOR_HANDLE AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE type, UINT count = 1)
	{
		return s_descriptorAllocators[type].Allocate(count);
	}

public:
	static Device* s_device;

private:
	static DescriptorAllocator* s_descriptorAllocators;

};

#endif // ! GRAPH



