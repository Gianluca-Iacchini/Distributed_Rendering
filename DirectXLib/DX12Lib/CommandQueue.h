#pragma once

#define OUT


#include "Fence.h"
#include "CommandAllocator.h"


namespace DX12Lib {

	class CommandList;
	class CommandAllocator;
	class Device;

	class CommandQueue
	{
		friend class CommandQueueManager;

	public:
		CommandQueue(D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT);
		~CommandQueue();

		void Create(Device& device);

		UINT64 ExecuteCommandLists(std::vector<CommandList*> cmdList);
		UINT64 ExecuteCommandList(CommandList& cmdList);


		void Flush();
		void WaitForFence(UINT64 fenceValue);
		UINT64 GetFenceValue() { return m_fence->CurrentFenceValue; }
		UINT64 GetGPUFenceValue() { return m_fence->GetGPUFenceValue(); }

		CommandAllocator* RequestAllocator();
		void DiscardAllocator(UINT64 fenceValue, CommandAllocator* allocator);

	private:
		UINT64 ExecuteAndSignal(std::vector<CommandList*> cmdLists);

		CommandAllocatorPool m_allocatorPool;

	private:
		Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_commandQueue;
		D3D12_COMMAND_LIST_TYPE m_type;

		std::unique_ptr<Fence> m_fence;
		std::mutex m_fenceMutex;

		std::vector<ID3D12CommandList*> m_executeCmdLists;

	public:
		ID3D12CommandQueue* Get() const { return m_commandQueue.Get(); }
		ID3D12CommandQueue** GetAddressOf() { return m_commandQueue.GetAddressOf(); }
		Microsoft::WRL::ComPtr<ID3D12CommandQueue> GetComPtr() const { return m_commandQueue; }
	};


	class CommandQueueManager
	{
	public:
		CommandQueueManager(Device& device);
		~CommandQueueManager();

		void Create();

		CommandQueue& GetGraphicsQueue() { return m_graphicsQueue; }
		CommandQueue& GetComputeQueue() { return m_computeQueue; }
		CommandQueue& GetCopyQueue() { return m_copyQueue; }
		CommandQueue& GetQueue(D3D12_COMMAND_LIST_TYPE type)
		{
			switch (type)
			{
			case D3D12_COMMAND_LIST_TYPE_DIRECT:
				return m_graphicsQueue;
			case D3D12_COMMAND_LIST_TYPE_COMPUTE:
				return m_computeQueue;
			case D3D12_COMMAND_LIST_TYPE_COPY:
				return m_copyQueue;
			default:
				return m_graphicsQueue;
			}
		};

		void CreateCommandList(D3D12_COMMAND_LIST_TYPE type, OUT CommandList** cmdList, OUT CommandAllocator** cmdAllocator);

	private:



		Device& m_device;

		CommandQueue m_graphicsQueue;
		CommandQueue m_computeQueue;
		CommandQueue m_copyQueue;
	};
}



