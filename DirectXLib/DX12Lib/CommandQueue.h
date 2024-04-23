#include "Helpers.h"
#include "Device.h"
#include "Fence.h"
#include "mutex"
#include "CommandAllocator.h"


#ifndef COMMAND_QUEUE_H
#define COMMAND_QUEUE_H

#define OUT

class CommandList;


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

	void CreateCommandList(D3D12_COMMAND_LIST_TYPE type, OUT CommandList** cmdList, OUT CommandAllocator** cmdAllocator);

private:



	Device& m_device;

	CommandQueue m_graphicsQueue;
	CommandQueue m_computeQueue;
	CommandQueue m_copyQueue;
};
#endif // !COMMAND_QUEUE_H



