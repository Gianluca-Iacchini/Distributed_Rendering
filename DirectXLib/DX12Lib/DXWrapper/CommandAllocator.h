#pragma once

#include <wrl/client.h>
#include <d3d12.h>
#include <mutex>
#include <queue>

namespace DX12Lib {

	class Device;

	class CommandAllocator
	{
	public:
		CommandAllocator(Device& device, D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT);
		~CommandAllocator();
		void Reset() { ThrowIfFailed(m_commandAllocator->Reset()); }


	private:
		Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_commandAllocator;

	public:
		ID3D12CommandAllocator* operator->() const { return m_commandAllocator.Get(); }
		ID3D12CommandAllocator* operator*() const { return m_commandAllocator.Get(); }
		Microsoft::WRL::ComPtr<ID3D12CommandAllocator>& operator&() { return m_commandAllocator; }

		ID3D12CommandAllocator* Get() const { return m_commandAllocator.Get(); }
		ID3D12CommandAllocator** GetAddressOf() { return m_commandAllocator.GetAddressOf(); }
		Microsoft::WRL::ComPtr<ID3D12CommandAllocator> GetComPtr() const { return m_commandAllocator; }
	};

	class CommandAllocatorPool
	{
	public:
		CommandAllocatorPool(D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT);
		~CommandAllocatorPool();


		CommandAllocator* RequestAllocator(UINT64 completedFenceValue);
		void DiscardAllocator(UINT64 fenceValue, CommandAllocator* allocator);

		inline size_t Size() const { return m_commandAllocatorPool.size(); }

	private:
		const D3D12_COMMAND_LIST_TYPE m_type;
		std::vector<CommandAllocator*> m_commandAllocatorPool;
		std::queue<std::pair<UINT64, CommandAllocator*>> m_availableCommandAllocators;
		std::mutex m_cmdAllocatorMutex;
	};

}



