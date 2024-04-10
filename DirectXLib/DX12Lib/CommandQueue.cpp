#include "CommandQueue.h"
#include "Device.h"
#include "CommandList.h"

using namespace Microsoft::WRL;

CommandQueue::CommandQueue(Device& device, D3D12_COMMAND_QUEUE_DESC cmdQueueDesc)
{
	ThrowIfFailed(device.GetComPtr()->CreateCommandQueue(&cmdQueueDesc, IID_PPV_ARGS(m_commandQueue.GetAddressOf())));
	m_fence = std::make_unique<Fence>(device, 0);
}

CommandQueue::CommandQueue(Device& device, D3D12_COMMAND_LIST_TYPE type)
{
	D3D12_COMMAND_QUEUE_DESC queueDesc = {};
	queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	queueDesc.NodeMask = 0;
	queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
	queueDesc.Type = type;

	ThrowIfFailed(device.GetComPtr()->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(m_commandQueue.GetAddressOf())));
	m_fence = std::make_unique<Fence>(device, 0);
}

void CommandQueue::ExecuteCommandLists(std::vector<CommandList> cmdLists)
{
	std::vector<ID3D12CommandList*> cmdListExecute;

	for (auto& cmdList : cmdLists)
	{
		cmdList.Close();
		cmdListExecute.push_back(cmdList.Get());
	}

	m_commandQueue->ExecuteCommandLists(static_cast<UINT>(cmdListExecute.size()), cmdListExecute.data());

}

void CommandQueue::ExecuteCommandList(CommandList& cmdList)
{
	ID3D12CommandList* cmdListExecute = cmdList.Get();

	cmdList.Close();

	m_commandQueue->ExecuteCommandLists(1, &cmdListExecute);
}

void CommandQueue::Signal()
{
	ThrowIfFailed(m_commandQueue->Signal(m_fence->Get(), m_fence->GetCPUFenceValue()));
}

void CommandQueue::Flush()
{
	m_fence->IncreaseCounter();
	this->Signal();

	m_fence->WaitForFence();
}
