#pragma once
#include "Helpers.h"
#include "CommandList.h"
#include "CommandAllocator.h"

#define MAX_RESOURCE_BARRIERS 16

class Resource;
class DescriptorHeap;

class CommandContext
{
	friend class CommandContextManager;

public:
	CommandContext(D3D12_COMMAND_LIST_TYPE type);
	~CommandContext();

	void Initialize();
	void Reset();

	void TransitionResource(Resource& res, D3D12_RESOURCE_STATES newState, bool transitionNow = false);
	void FlushResourceBarriers();
	void BindDescriptorHeaps(DescriptorHeap heap);

	static void CommitGraphicsResources(D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT);
	static void InitializeTexture(Resource& dest, UINT numSubresources, D3D12_SUBRESOURCE_DATA subresources[]);



	UINT64 Flush(bool waitForCompletion = false);
	UINT64 Finish(bool waitForCompletion = false);

public:
	CommandAllocator* m_currentAllocator = nullptr;
	CommandList* m_commandList = nullptr;

private:
	CommandContext(const CommandContext&) = delete;
	CommandContext& operator=(const CommandContext&) = delete;

protected:
	D3D12_COMMAND_LIST_TYPE m_type;
	D3D12_RESOURCE_BARRIER m_resourceBarrier[MAX_RESOURCE_BARRIERS];
	UINT m_numBarriersToFlush = 0;
};


class CommandContextManager
{
public:
	CommandContextManager() {};
	~CommandContextManager() { DestroyAllContexts(); };
	CommandContext* AllocateContext(D3D12_COMMAND_LIST_TYPE type);
	void FreeContext(CommandContext* usedContext);
	void DestroyAllContexts();

private:
	std::vector<std::unique_ptr<CommandContext>> m_contextPool[4];
	std::queue<CommandContext*> m_availableContexts[4];
	std::mutex m_contextAllocationMutex;
};
