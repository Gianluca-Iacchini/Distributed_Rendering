#include "CommandQueue.h"
#include "Device.h"
#include "CommandList.h"

using namespace Microsoft::WRL;

CommandQueue::CommandQueue(D3D12_COMMAND_LIST_TYPE type) : m_type(type)
{
}

CommandQueue::~CommandQueue()
{
	m_executeCmdLists.clear();
}

void CommandQueue::Create(Device& device)
{
	D3D12_COMMAND_QUEUE_DESC queueDesc = {};
	queueDesc.NodeMask = 0;
	queueDesc.Type = m_type;

	ThrowIfFailed(device.GetComPtr()->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(m_commandQueue.GetAddressOf())));
	m_fence = std::make_unique<Fence>(device, 0, 1);
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

	// Flush the command queue, release the lock before the call to wait for fence since the fence
	// will lock again on its own
	{
		std::lock_guard<std::mutex> lock(m_fenceMutex);

		m_fence->CurrentFenceValue += 1;

		ThrowIfFailed(m_commandQueue->Signal(m_fence->Get(), m_fence->CurrentFenceValue));
	}
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

CommandQueueManager::CommandQueueManager(Device& device)
	: m_device(device),
	m_graphicsQueue(D3D12_COMMAND_LIST_TYPE_DIRECT),
	m_computeQueue(D3D12_COMMAND_LIST_TYPE_COMPUTE),
	m_copyQueue(D3D12_COMMAND_LIST_TYPE_COPY)
{
}

CommandQueueManager::~CommandQueueManager()
{
}

void CommandQueueManager::Create()
{
	m_graphicsQueue.Create(m_device);
	m_computeQueue.Create(m_device);
	m_copyQueue.Create(m_device);
}
