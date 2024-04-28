#ifndef  GRAPHICS_CORE_H
#define GRAPHICS_CORE_H
#include "DX12Lib/DescriptorHeap.h"
#include "GraphicsMemory.h"
#include "Helpers.h"
#include "Mouse.h"

class Device;
class CommandAllocatorPool;
class CommandQueueManager;
class CommandAllocator;
class CommandList;
class CommandContextManager;

namespace Graphics
{
	extern DescriptorAllocator s_descriptorAllocators[];
	extern std::shared_ptr<Device> s_device;
	extern std::unique_ptr<CommandQueueManager> s_commandQueueManager;
	extern std::unique_ptr<CommandContextManager> s_commandContextManager;
	extern std::unique_ptr<DirectX::GraphicsMemory> s_graphicsMemory;
	extern Microsoft::WRL::ComPtr<ID3D12DeviceRemovedExtendedDataSettings1> s_dredSettings;

	extern std::unique_ptr<DirectX::Mouse> s_mouse;

	bool Initialize();
	void Shutdown();
	void DeviceRemovedHandler();

	inline D3D12_CPU_DESCRIPTOR_HANDLE AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE type, UINT count = 1)
	{
		return s_descriptorAllocators[type].Allocate(count);
	}

};

#endif // ! GRAPH



