#pragma once

#include "DX12Lib/DXWrapper/DescriptorHeap.h"
#include "GraphicsMemory.h"
#include "Mouse.h"
#include "Keyboard.h"
#include "DX12Lib/DXWrapper/Texture.h"
#include "DX12Lib/Models/MaterialManager.h"
#include "Renderer.h"



namespace DX12Lib {
	class Device;
	class CommandAllocatorPool;
	class CommandQueueManager;
	class CommandAllocator;
	class CommandList;
	class CommandContextManager;
	class PipelineState;
	class Shader;

}

namespace Graphics
{
	extern DX12Lib::DescriptorAllocator s_descriptorAllocators[];
	extern std::shared_ptr<DX12Lib::Device> s_device;
	extern std::unique_ptr<DX12Lib::CommandQueueManager> s_commandQueueManager;
	extern std::unique_ptr<DX12Lib::CommandContextManager> s_commandContextManager;
	extern Microsoft::WRL::ComPtr<ID3D12DeviceRemovedExtendedDataSettings1> s_dredSettings;

	extern std::unique_ptr<DirectX::Mouse> s_mouse;
	extern std::unique_ptr<DirectX::Keyboard> s_keyboard;
	extern std::unique_ptr<DirectX::Keyboard::KeyboardStateTracker> s_kbTracker;


	extern DXGI_FORMAT m_backBufferFormat;
	extern DXGI_FORMAT m_depthStencilFormat;

	bool Initialize();
	void Shutdown();
	void DeviceRemovedHandler();

	inline D3D12_CPU_DESCRIPTOR_HANDLE AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE type, UINT count = 1)
	{
		return s_descriptorAllocators[type].Allocate(count);
	}

};




