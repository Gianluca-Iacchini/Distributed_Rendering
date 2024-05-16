#pragma once

#include "DX12Lib/DXWrapper/DescriptorHeap.h"
#include "GraphicsMemory.h"
#include "Mouse.h"
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
	extern std::unique_ptr<DirectX::GraphicsMemory> s_graphicsMemory;
	extern Microsoft::WRL::ComPtr<ID3D12DeviceRemovedExtendedDataSettings1> s_dredSettings;
	extern std::unique_ptr<DX12Lib::TextureManager> s_textureManager;
	extern std::unique_ptr<DX12Lib::MaterialManager> s_materialManager;
	extern std::unordered_map<std::wstring, std::shared_ptr<DX12Lib::PipelineState>> s_PSOs;
	extern std::unordered_map<std::wstring, std::shared_ptr<DX12Lib::Shader>> s_shaders;

	extern std::unique_ptr<DirectX::Mouse> s_mouse;

	extern std::shared_ptr<DX12Lib::DescriptorHeap> s_textureHeap;

	extern Renderer s_renderer;

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




