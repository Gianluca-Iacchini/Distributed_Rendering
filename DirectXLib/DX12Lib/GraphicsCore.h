#pragma once

#include "DX12Lib/DescriptorHeap.h"
#include "GraphicsMemory.h"
#include "Mouse.h"
#include "Texture.h"
#include "MaterialManager.h"


namespace DX12Lib {
	class Device;
	class CommandAllocatorPool;
	class CommandQueueManager;
	class CommandAllocator;
	class CommandList;
	class CommandContextManager;
}

namespace Graphics
{
	extern DX12Lib::DescriptorAllocator s_descriptorAllocators[];
	extern std::shared_ptr<DX12Lib::Device> s_device;
	extern std::unique_ptr<DX12Lib::CommandQueueManager> s_commandQueueManager;
	extern std::unique_ptr<DX12Lib::CommandContextManager> s_commandContextManager;
	extern std::unique_ptr<DirectX::GraphicsMemory> s_graphicsMemory;
	extern Microsoft::WRL::ComPtr<ID3D12DeviceRemovedExtendedDataSettings1> s_dredSettings;
	extern std::unique_ptr<DX12Lib::TextureManager> s_textureManager;
	extern std::unique_ptr<DX12Lib::MaterialManager> s_materialManager;

	extern std::unique_ptr<DirectX::Mouse> s_mouse;

	bool Initialize();
	void Shutdown();
	void DeviceRemovedHandler();

	inline D3D12_CPU_DESCRIPTOR_HANDLE AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE type, UINT count = 1)
	{
		return s_descriptorAllocators[type].Allocate(count);
	}

};




