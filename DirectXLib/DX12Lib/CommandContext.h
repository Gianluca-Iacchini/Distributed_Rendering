#pragma once
#include "Helpers.h"
#include "CommandList.h"
#include "CommandAllocator.h"

#define MAX_RESOURCE_BARRIERS 16

class Resource;

class CommandContext
{

public:
	CommandContext(D3D12_COMMAND_LIST_TYPE type);
	~CommandContext();

	void Initialize();

	void TransitionResource(Resource& res, D3D12_RESOURCE_STATES newState, bool transitionNow);
	void FlushResourceBarriers();

	void Reset();

	UINT64 Finish();

public:
	CommandAllocator* m_currentAllocator = nullptr;
	CommandList* m_commandList = nullptr;

protected:
	D3D12_RESOURCE_BARRIER m_resourceBarrier[MAX_RESOURCE_BARRIERS];
	UINT m_numBarriersToFlush = 0;
};

