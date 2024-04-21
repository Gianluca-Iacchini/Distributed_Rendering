#include "CommandQueue.h"
#include "Device.h"
#include "CommandList.h"

using namespace Microsoft::WRL;

CommandQueue::CommandQueue(Device& device, D3D12_COMMAND_QUEUE_DESC cmdQueueDesc)
{
	ThrowIfFailed(device.GetComPtr()->CreateCommandQueue(&cmdQueueDesc, IID_PPV_ARGS(m_commandQueue.GetAddressOf())));
	m_fence = std::make_unique<Fence>(device, 0, 1);
}

CommandQueue::CommandQueue(Device& device, D3D12_COMMAND_LIST_TYPE type)
{
	D3D12_COMMAND_QUEUE_DESC queueDesc = {};
	queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	queueDesc.NodeMask = 0;
	queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
	queueDesc.Type = type;

	ThrowIfFailed(device.GetComPtr()->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(m_commandQueue.GetAddressOf())));
	m_fence = std::make_unique<Fence>(device, 0, 1);
}

CommandQueue::~CommandQueue()
{
	m_executeCmdLists.clear();
}

UINT64 CommandQueue::ExecuteCommandLists(std::vector<CommandList*> cmdLists)
{
	return this->ExecuteAndSignal(cmdLists);
}

UINT64 CommandQueue::ExecuteCommandList(CommandList& cmdList)
{
	std::vector<CommandList*> cmdLists = { &cmdList };

	return this->ExecuteAndSignal(cmdLists);
}

void CommandQueue::Flush()
{
	m_fence->CurrentFenceValue += 1;

	ThrowIfFailed(m_commandQueue->Signal(m_fence->Get(), m_fence->CurrentFenceValue));

	m_fence->WaitForCurrentFence();
}


void CommandQueue::WaitForFence(UINT64 fenceValue)
{
	m_fence->WaitForFence(fenceValue);
}

UINT64 CommandQueue::ExecuteAndSignal(std::vector<CommandList*> cmdLists)
{
	assert (cmdLists.size() > 0);

	std::lock_guard<std::mutex> lock(m_fenceMutex);

	UINT cmdListSize = cmdLists.size();

	m_executeCmdLists.resize(cmdListSize);

	for (UINT i = 0; i < cmdListSize; i++)
	{
		cmdLists[i]->Close();
		m_executeCmdLists[i] = cmdLists[i]->Get();
	}

	m_commandQueue->ExecuteCommandLists(cmdListSize, m_executeCmdLists.data());

	ThrowIfFailed(m_commandQueue->Signal(m_fence->Get(), m_fence->CurrentFenceValue));


	return m_fence->CurrentFenceValue++;
}

