#include "DX12Lib/pch.h"

#include "CommandContext.h"
#include "DX12Lib/DXWrapper/Resource.h"
#include "GraphicsMemory.h"
#include "DX12Lib/DXWrapper/QueryHeap.h"
#include "DX12Lib/DXWrapper/GPUBuffer.h"


using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;

CommandContext::CommandContext(D3D12_COMMAND_LIST_TYPE type)
	: m_type(type), m_resourceBarrier{}
{

	ZeroMemory(m_currentDescriptorHeaps, sizeof(m_currentDescriptorHeaps));
	m_currentAllocator = nullptr;
	m_commandList = nullptr;
	m_currentPipelineState = nullptr;
	m_numBarriersToFlush = 0;
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

void CommandContext::InitializeApp()
{
	s_commandQueueManager->CreateCommandList(m_type, &m_commandList, &m_currentAllocator);
}

CommandContext& DX12Lib::CommandContext::Begin()
{
	CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);

	assert(context != nullptr && "Context is null");

	return *context;
}

void DX12Lib::CommandContext::CopyBuffer(Resource& dest, Resource& src)
{
	TransitionResource(dest, D3D12_RESOURCE_STATE_COPY_DEST);
	TransitionResource(src, D3D12_RESOURCE_STATE_COPY_SOURCE);
	FlushResourceBarriers();
	m_commandList->Get()->CopyResource(dest.Get(), src.Get());
}

void DX12Lib::CommandContext::CopyBufferRegion(Resource& dest, size_t destOffset, Resource& src, size_t srcOffset, size_t numBytes)
{
	TransitionResource(dest, D3D12_RESOURCE_STATE_COPY_DEST);
	TransitionResource(src, D3D12_RESOURCE_STATE_COPY_SOURCE);
	FlushResourceBarriers();
	m_commandList->Get()->CopyBufferRegion(dest.Get(), destOffset, src.Get(), srcOffset, numBytes);
}

void DX12Lib::CommandContext::SetPipelineState(PipelineState* pipelineState)
{
	if (m_currentPipelineState != pipelineState)
	{
		m_currentPipelineState = pipelineState;
		pipelineState->Use(*m_commandList);

		m_currentPipelineState->UseRootSignature(*m_commandList);
	}
}

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

void DX12Lib::CommandContext::TransitionResource(ID3D12Resource* res, D3D12_RESOURCE_STATES beforeState, D3D12_RESOURCE_STATES newState, bool transitionNow)
{
	if (beforeState != newState)
	{
		assert(m_numBarriersToFlush < MAX_RESOURCE_BARRIERS && "Exceeded maximum number of resource barriers");
		D3D12_RESOURCE_BARRIER& barrier = m_resourceBarrier[m_numBarriersToFlush++];
		barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barrier.Transition.pResource = res;
		barrier.Transition.StateBefore = beforeState;
		barrier.Transition.StateAfter = newState;
		barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
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

void DX12Lib::CommandContext::InsertUAVBarrier(Resource* resource, bool flushImmediate)
{
	D3D12_RESOURCE_BARRIER barrierDesc = {};
	barrierDesc.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
	barrierDesc.UAV.pResource = (resource == nullptr ? nullptr : resource->Get());
	m_commandList->Get()->ResourceBarrier(1, &barrierDesc);
	if (flushImmediate)
	{
		FlushResourceBarriers();
	}
}

void DX12Lib::CommandContext::AddUAVIfNoBarriers(Resource* resource, bool flushImmediate)
{
	if (m_numBarriersToFlush == 0)
	{
		InsertUAVBarrier(resource, false);
	}

	if (flushImmediate)
	{
		FlushResourceBarriers();
	}
}

void CommandContext::InitializeTexture(Resource& dest, UINT numSubresources, D3D12_SUBRESOURCE_DATA subresources[])
{
	UINT64 uploadBufferSize = GetRequiredIntermediateSize(dest.Get(), 0, numSubresources);

	auto context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);

	DirectX::GraphicsResource uploadBuffer = Renderer::s_graphicsMemory->Allocate(uploadBufferSize);
	
	UpdateSubresources(context->m_commandList->Get(), dest.Get(), uploadBuffer.Resource(), 0, 0, numSubresources, subresources);

	context->TransitionResource(dest, D3D12_RESOURCE_STATE_GENERIC_READ);

	context->Finish(true);
}


void DX12Lib::CommandContext::SetDescriptorHeap(DescriptorHeap* heap)
{

	if (heap != nullptr && m_currentDescriptorHeaps[heap->GetType()] != heap)
	{
		m_currentDescriptorHeaps[heap->GetType()] = heap;
		BindDescriptorHeaps();
	}
}

void DX12Lib::CommandContext::SetDescriptorHeaps(std::vector<DescriptorHeap*> heaps)
{
	bool heapsChanged = false;

	for (auto* heap : heaps)
	{
		if (heap != nullptr && m_currentDescriptorHeaps[heap->GetType()] != heap)
		{
			m_currentDescriptorHeaps[heap->GetType()] = heap;
			heapsChanged = true;
		}
	}

	if (heapsChanged)
	{
		BindDescriptorHeaps();
	}
}

void DX12Lib::CommandContext::BindDescriptorHeaps()
{
	UINT NonNullHeaps = 0;
	DescriptorHeap* HeapsToBind[D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES];
	for (UINT i = 0; i < D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES; ++i)
	{
		DescriptorHeap* HeapIter = m_currentDescriptorHeaps[i];
		if (HeapIter != nullptr)
			HeapsToBind[NonNullHeaps++] = HeapIter;
	}

	if (NonNullHeaps > 0)
		m_commandList->SetDescriptorHeaps(HeapsToBind, NonNullHeaps);
}

void DX12Lib::CommandContext::BeginQuery(QueryHeap& queryHeap, QueryHandle& handle, int offset)
{
	// Only timestamp queries are supported
	if (queryHeap.GetType() != D3D12_QUERY_HEAP_TYPE_TIMESTAMP)
		return;

	m_commandList->Get()->BeginQuery(queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, handle.GetIndex() + offset);

	handle.QueryStarted();
}

void DX12Lib::CommandContext::EndQuery(QueryHeap& queryHeap, QueryHandle& handle, int offset)
{
	// Only timestamp queries are supported
	if (queryHeap.GetType() != D3D12_QUERY_HEAP_TYPE_TIMESTAMP)
		return;

	m_commandList->Get()->EndQuery(queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, handle.GetIndex() + offset);

	if (queryHeap.GetType() != D3D12_QUERY_HEAP_TYPE_TIMESTAMP)
		handle.QueryEnded();
}

void DX12Lib::CommandContext::ResolveQueryData(QueryHeap& queryHeap, QueryHandle& handle, ReadBackBuffer& destBuffer, UINT numQueries)
{
	if (queryHeap.GetType() != D3D12_QUERY_HEAP_TYPE_TIMESTAMP)
		return;

	assert(handle.IsValid() && "Invalid query handle");

	UINT queryToResolve = numQueries == 0 ? handle.GetSize() : numQueries;

	this->TransitionResource(destBuffer, D3D12_RESOURCE_STATE_COPY_DEST, true);
	m_commandList->Get()->ResolveQueryData(queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, handle.GetIndex(), queryToResolve, destBuffer.Get(), 0);
}

void CommandContext::Reset()
{
	m_currentAllocator = s_commandQueueManager->GetQueue(m_type).RequestAllocator();
	m_commandList->Reset(*m_currentAllocator);
	m_numBarriersToFlush = 0;
	m_currentPipelineState = nullptr;

	BindDescriptorHeaps();
}

UINT64 CommandContext::Flush(bool waitForCompletion)
{
	FlushResourceBarriers();

	CommandQueue& queue = s_commandQueueManager->GetQueue(m_type);

	UINT64 fenceValue = queue.ExecuteCommandList(*m_commandList);

	if (waitForCompletion)
		queue.WaitForFence(fenceValue);

	
	m_commandList->Reset(*m_currentAllocator);


	if (m_currentPipelineState != nullptr)
	{
		m_currentPipelineState->Use(*m_commandList);
		m_currentPipelineState->UseRootSignature(*m_commandList);
	}

	BindDescriptorHeaps();
	
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
		context->InitializeApp();
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

void DX12Lib::GraphicsContext::ClearColor(ColorBuffer& target, D3D12_RECT* rect)
{
	FlushResourceBarriers();
	m_commandList->Get()->ClearRenderTargetView(target.GetRTV(), target.GetClearColor().GetPtr(), rect == nullptr ? 0 : 1, rect);
}

void DX12Lib::GraphicsContext::ClearColor(ColorBuffer& target, float color[4], D3D12_RECT* rect)
{
	FlushResourceBarriers();
	m_commandList->Get()->ClearRenderTargetView(target.GetRTV(), color, rect == nullptr ? 0 : 1, rect);
}

void DX12Lib::GraphicsContext::ClearDepth(DepthBuffer& depthBuffer)
{
	FlushResourceBarriers();
	m_commandList->Get()->ClearDepthStencilView(depthBuffer.GetDSV(), D3D12_CLEAR_FLAG_DEPTH, depthBuffer.GetClearDepth(), depthBuffer.GetClearStencil(), 0, nullptr);
}

void DX12Lib::GraphicsContext::ClearDepthAndStencil(DepthBuffer& depthBuffer)
{
	FlushResourceBarriers();
	m_commandList->Get()->ClearDepthStencilView(depthBuffer.GetDSV(), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, depthBuffer.GetClearDepth(), depthBuffer.GetClearStencil(), 0, nullptr);
}

void DX12Lib::GraphicsContext::SetRenderTargets(UINT numRTVs, const D3D12_CPU_DESCRIPTOR_HANDLE rtvs[], D3D12_CPU_DESCRIPTOR_HANDLE dsv)
{
	m_commandList->Get()->OMSetRenderTargets(numRTVs, rtvs, FALSE, &dsv);
}

void DX12Lib::GraphicsContext::SetDepthStencilTarget(D3D12_CPU_DESCRIPTOR_HANDLE dsv)
{
	SetRenderTargets(0, nullptr, dsv);
}

void DX12Lib::GraphicsContext::SetViewportAndScissor(D3D12_VIEWPORT& viewport, D3D12_RECT& scissorRect)
{
	assert(scissorRect.left < scissorRect.right && scissorRect.top < scissorRect.bottom);

	m_commandList->Get()->RSSetViewports(1, &viewport);
	m_commandList->Get()->RSSetScissorRects(1, &scissorRect);
}



void GraphicsContext::CommitGraphicsResources(D3D12_COMMAND_LIST_TYPE type)
{
	Renderer::s_graphicsMemory->Commit(s_commandQueueManager->GetQueue(type).Get());
}

GraphicsContext& DX12Lib::GraphicsContext::Begin()
{
	CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_DIRECT);

	assert(context != nullptr && "Context is null");

	return reinterpret_cast<GraphicsContext&>(*context);
}

ComputeContext& DX12Lib::ComputeContext::Begin()
{
	CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_COMPUTE);

	assert(context != nullptr && "Context is null");

	return reinterpret_cast<ComputeContext&>(*context);
}

ComputeContext* DX12Lib::ComputeContext::BeginPtr()
{
	CommandContext* context = s_commandContextManager->AllocateContext(D3D12_COMMAND_LIST_TYPE_COMPUTE);

	assert(context != nullptr && "Context is null");

	return reinterpret_cast<ComputeContext*>(context);
}

void DX12Lib::ComputeContext::Dispatch(size_t groupCountX, size_t groupCountY, size_t groupCountZ)
{
	FlushResourceBarriers();
	m_commandList->Get()->Dispatch((UINT)groupCountX, (UINT)groupCountY, (UINT)groupCountZ);
}

void DX12Lib::ComputeContext::Dispatch1D(size_t threadCount, size_t groupSize)
{
	Dispatch(MathHelper::DivideByMultiple(threadCount, groupSize), 1, 1);
}

void DX12Lib::ComputeContext::Dispatch2D(size_t threadCountX, size_t threadCountY, size_t groupSizeX, size_t groupSizeY)
{
	Dispatch(MathHelper::DivideByMultiple(threadCountX, groupSizeX), 
			 MathHelper::DivideByMultiple(threadCountY, groupSizeY), 
			 1);
}

void DX12Lib::ComputeContext::Dispatch3D(size_t threadCountX, size_t threadCountY, size_t threadCountZ, size_t groupSizeX, size_t groupSizeY, size_t groupSizeZ)
{
	Dispatch(MathHelper::DivideByMultiple(threadCountX, groupSizeX),
			 MathHelper::DivideByMultiple(threadCountY, groupSizeY),
			 MathHelper::DivideByMultiple(threadCountZ, groupSizeZ));
}

