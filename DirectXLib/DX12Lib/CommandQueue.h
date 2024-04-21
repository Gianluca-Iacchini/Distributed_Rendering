#include "Helpers.h"
#include "Device.h"
#include "Fence.h"
#include "mutex"

#ifndef COMMAND_QUEUE_H
#define COMMAND_QUEUE_H

class CommandList;


class CommandQueue
{
public:
	CommandQueue(Device& device, D3D12_COMMAND_QUEUE_DESC cmdQueueDesc);
	CommandQueue(Device& device, D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT);
	~CommandQueue();

	//bool Initialize(D3D12_COMMAND_LIST_TYPE type, bool isBundle = false);
	UINT64 ExecuteCommandLists(std::vector<CommandList*> cmdList);
	UINT64 ExecuteCommandList(CommandList& cmdList);
	//void ExecuteCommandList(ID3D12CommandList* commandList);

	//void Wait(ID3D12Fence* fence, UINT64 fenceValue);
	void Flush();

	void WaitForFence(UINT64 fenceValue);

	UINT64 GetFenceValue() { return m_fence->CurrentFenceValue; }

private:
	UINT64 ExecuteAndSignal(std::vector<CommandList*> cmdLists);

	Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_commandQueue;
	std::unique_ptr<Fence> m_fence;
	HANDLE m_fenceEvent;
	std::mutex m_fenceMutex;

	std::vector<ID3D12CommandList*> m_executeCmdLists;

public:
	ID3D12CommandQueue* Get() const { return m_commandQueue.Get(); }
	ID3D12CommandQueue** GetAddressOf() { return m_commandQueue.GetAddressOf(); }
	Microsoft::WRL::ComPtr<ID3D12CommandQueue> GetComPtr() const { return m_commandQueue; }
};
#endif // !COMMAND_QUEUE_H



