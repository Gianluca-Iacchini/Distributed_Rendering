#pragma once
#include "Helpers.h"
#include "CommandList.h"
#include "CommandAllocator.h"


class CommandContext
{

public:
	CommandContext(D3D12_COMMAND_LIST_TYPE type);
	~CommandContext();

	void Initialize();

	void Reset();

	void Finish();

public:
	CommandAllocator* m_currentAllocator = nullptr;
	CommandList* m_commandList = nullptr;
};

