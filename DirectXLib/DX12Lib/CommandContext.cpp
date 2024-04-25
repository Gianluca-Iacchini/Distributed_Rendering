#include "CommandContext.h"
#include "CommandQueue.h"
#include "CommandList.h"
#include "CommandAllocator.h"
#include "GraphicsCore.h"
#include "Resource.h"


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

#include <iostream>

void CommandContext::TransitionResource(Resource& res, D3D12_RESOURCE_STATES newState, bool transitionNow)
{
	D3D12_RESOURCE_STATES oldState = res.m_currentState;

	if (oldState != newState)
	{
		assert(m_numBarriersToFlush < MAX_RESOURCE_BARRIERS && "Exceeded maximum number of resource barriers");
		D3D12_RESOURCE_BARRIER& barrier = m_resourceBarrier[m_numBarriersToFlush++];
		barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barrier.Transition.pResource = res.Get();
		barrier.Transition.StateBefore = oldState;
		barrier.Transition.StateAfter = newState;
		barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

		// Not sure if i will need split barriers, so i'll keep this just in case
		if (newState == res.m_nextState)
		{
			barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_END_ONLY;
			res.m_nextState = (D3D12_RESOURCE_STATES)-1;
		}
		else
		{
			barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
		}

		res.m_currentState = newState;
	}

	if (transitionNow || m_numBarriersToFlush == MAX_RESOURCE_BARRIERS)
	{
		FlushResourceBarriers();
	}
}

void CommandContext::FlushResourceBarriers()
{
	if (m_numBarriersToFlush > 0)
	{
		m_commandList->GetComPtr()->ResourceBarrier(m_numBarriersToFlush, m_resourceBarrier);
		m_numBarriersToFlush = 0;
	}
}

void CommandContext::Reset()
{
	m_currentAllocator = Graphics::s_commandQueueManager->GetGraphicsQueue().RequestAllocator();
	m_commandList->Reset(*m_currentAllocator);

}

UINT64 CommandContext::Finish()
{
	CommandQueue& queue = Graphics::s_commandQueueManager->GetGraphicsQueue();

	UINT64 fenceValue = queue.ExecuteCommandList(*m_commandList);
	queue.DiscardAllocator(fenceValue, m_currentAllocator);
	m_currentAllocator = nullptr;

	return fenceValue;
}

