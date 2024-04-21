#ifndef  GRAPHICS_CORE_H
#define GRAPHICS_CORE_H
#include "DX12Lib/DescriptorHeap.h"

class Device;
class CommandAllocatorPool;
class CommandQueueManager;
class CommandAllocator;
class CommandList;

namespace Graphics
{
	extern DescriptorAllocator s_descriptorAllocators[];
	extern std::shared_ptr<Device> s_device;
	extern std::unique_ptr<CommandAllocatorPool> s_commandAllocatorPool;
	extern std::unique_ptr<CommandQueueManager> s_commandQueueManager;
	extern CommandAllocator* s_initCommandAllocator;
	extern std::unique_ptr<CommandList> s_commandList;

	bool Initialize();
	void Shutdown();

	inline D3D12_CPU_DESCRIPTOR_HANDLE AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE type, UINT count = 1)
	{
		return s_descriptorAllocators[type].Allocate(count);
	}

};

#endif // ! GRAPH



