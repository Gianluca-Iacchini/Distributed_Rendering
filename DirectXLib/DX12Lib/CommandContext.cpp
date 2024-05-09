#include "pch.h"

#include "CommandContext.h"
#include "CommandQueue.h"
#include "CommandList.h"
#include "CommandAllocator.h"
#include "Resource.h"
#include "GraphicsMemory.h"


using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;

CommandContext::CommandContext(D3D12_COMMAND_LIST_TYPE type)
	: m_type(type), m_resourceBarrier{}
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
		s_commandQueueManager->GetQueue(m_type).DiscardAllocator(0, m_currentAllocator);
	}
}

void CommandContext::Initialize()
{
	s_commandQueueManager->CreateCommandList(m_type, &m_commandList, &m_currentAllocator);
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

void CommandContext::BindDescriptorHeaps(DescriptorHeap heap)
{
	ID3D12DescriptorHeap* heaps[] = { heap.Get() };
	m_commandList->GetComPtr()->SetDescriptorHeaps(1, heaps);
}

void CommandContext::CommitGraphicsResources(D3D12_COMMAND_LIST_TYPE type)
{
	s_graphicsMemory->Commit(s_commandQueueManager->GetQueue(type).Get());
}

void CommandContext::InitializeTexture(Resource& dest, UINT numSubresources, D3D12_SUBRESOURCE_DATA subresources[])
{
	UINT64 uploadBufferSize = GetRequiredIntermediateSize(dest.Get(), 0, numSubresources);

	auto context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);

	DirectX::GraphicsResource uploadBuffer = s_graphicsMemory->Allocate(uploadBufferSize);
	
	UpdateSubresources(context->m_commandList->Get(), dest.Get(), uploadBuffer.Resource(), 0, 0, numSubresources, subresources);

	context->TransitionResource(dest, D3D12_RESOURCE_STATE_GENERIC_READ);

	context->Finish(true);
}

void CommandContext::Reset()
{
	m_currentAllocator = s_commandQueueManager->GetQueue(m_type).RequestAllocator();
	m_commandList->Reset(*m_currentAllocator);
	m_numBarriersToFlush = 0;
}

UINT64 CommandContext::Flush(bool waitForCompletion)
{
	FlushResourceBarriers();

	CommandQueue& queue = s_commandQueueManager->GetQueue(m_type);

	UINT64 fenceValue = queue.ExecuteCommandList(*m_commandList);

	if (waitForCompletion)
		queue.WaitForFence(fenceValue);

	m_commandList->Reset(*m_currentAllocator);

	return fenceValue;
}

UINT64 CommandContext::Finish(bool waitForCompletion)
{
	FlushResourceBarriers();

	CommandQueue& queue = s_commandQueueManager->GetQueue(m_type);

	UINT64 fenceValue = queue.ExecuteCommandList(*m_commandList);
	queue.DiscardAllocator(fenceValue, m_currentAllocator);
	m_currentAllocator = nullptr;

	if (waitForCompletion)
		queue.WaitForFence(fenceValue);

	s_commandContextManager->FreeContext(this);

	return fenceValue;
}

CommandContext* CommandContextManager::AllocateContext(D3D12_COMMAND_LIST_TYPE type)
{
	std::lock_guard<std::mutex> lock(m_contextAllocationMutex);

	auto& availableContexts = m_availableContexts[type];
	
	CommandContext* context = nullptr;

	if (availableContexts.empty())
	{
		context = new CommandContext(type);
		m_contextPool[type].emplace_back(context);
		context->Initialize();
		return context;
	}

	context = availableContexts.front();
	availableContexts.pop();

	assert(context != nullptr && "Context is null");

	context->Reset();

	return context;
}

void CommandContextManager::FreeContext(CommandContext* usedContext)
{
	assert(usedContext != nullptr && "Used context is null");
	std::lock_guard<std::mutex> lock(m_contextAllocationMutex);
	m_availableContexts[usedContext->m_type].push(usedContext);
}

void CommandContextManager::DestroyAllContexts()
{
	// LinearAllocator DestroyAll
	// DynamicDescriptorHeap DestroyAll
	
	for (uint32_t i = 0; i < 4; ++i)
	{
		m_contextPool[i].clear();
	}

}
