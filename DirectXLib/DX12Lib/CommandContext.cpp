#include "CommandContext.h"
#include "CommandQueue.h"
#include "CommandList.h"
#include "CommandAllocator.h"
#include "GraphicsCore.h"
#include <iostream>


CommandContext::CommandContext(D3D12_COMMAND_LIST_TYPE type)
{
	m_currentAllocator = nullptr;
	m_commandList = nullptr;
}

CommandContext::~CommandContext()
{
	if (m_commandList != nullptr)
		delete m_commandList;
	
	if (m_currentAllocator != nullptr)
	{
		Graphics::s_commandQueueManager->GetGraphicsQueue().DiscardAllocator(0, m_currentAllocator);
	}
}

void CommandContext::Initialize()
{
	Graphics::s_commandQueueManager->CreateCommandList(D3D12_COMMAND_LIST_TYPE_DIRECT, &m_commandList, &m_currentAllocator);
}

void CommandContext::Reset()
{
	m_currentAllocator = Graphics::s_commandQueueManager->GetGraphicsQueue().RequestAllocator();
	m_commandList->Reset(*m_currentAllocator);

}

void CommandContext::Finish()
{
	CommandQueue& queue = Graphics::s_commandQueueManager->GetGraphicsQueue();
	UINT64 fenceValue = queue.ExecuteCommandList(*m_commandList);

	queue.DiscardAllocator(fenceValue, m_currentAllocator);
	m_currentAllocator = nullptr;
}

