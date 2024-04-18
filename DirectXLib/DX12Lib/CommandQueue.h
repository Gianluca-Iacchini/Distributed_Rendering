#include "Helpers.h"
#include "Device.h"
#include "Fence.h"

#ifndef COMMAND_QUEUE_H
#define COMMAND_QUEUE_H

class CommandList;


class CommandQueue
{
public:
	CommandQueue(Device& device, D3D12_COMMAND_QUEUE_DESC cmdQueueDesc);
	CommandQueue(Device& device, D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT);
	~CommandQueue() {};

	//bool Initialize(D3D12_COMMAND_LIST_TYPE type, bool isBundle = false);
	void ExecuteCommandLists(std::vector<CommandList> cmdList);
	void ExecuteCommandList(CommandList& cmdList);
	//void ExecuteCommandList(ID3D12CommandList* commandList);
	void Signal(Fence* fence, UINT64 value);
	void Signal(Fence* fence);
	//void Wait(ID3D12Fence* fence, UINT64 fenceValue);
	void Flush(Fence* fence);


private:
	Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_commandQueue;

public:
	ID3D12CommandQueue* Get() const { return m_commandQueue.Get(); }
	ID3D12CommandQueue** GetAddressOf() { return m_commandQueue.GetAddressOf(); }
	Microsoft::WRL::ComPtr<ID3D12CommandQueue> GetComPtr() const { return m_commandQueue; }
};
#endif // !COMMAND_QUEUE_H



