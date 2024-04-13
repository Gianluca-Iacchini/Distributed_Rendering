#include "CommandList.h"
#include "CommandAllocator.h"
#include "PipelineState.h"
#include "Device.h"


CommandList::CommandList(Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList) : m_commandList(commandList)
{
}

CommandList::CommandList(Device& device, int frameCount, D3D12_COMMAND_LIST_TYPE cmdType, PipelineState* pipelineState)
	: m_nFrameCount(frameCount), m_cmdType(cmdType)
{
	ID3D12PipelineState* pso = nullptr;
	if (pipelineState != nullptr)
	{
		pso = pipelineState->GetComPtr().Get();
	}

	CreateCmdAllocators(device);
	ThrowIfFailed(device.GetComPtr()->CreateCommandList(0, cmdType, v_cmdAllocators[m_currentFrame]->Get(), pso, IID_PPV_ARGS(m_commandList.GetAddressOf())));
}

void CommandList::Reset(PipelineState* pipelineState)
{
	ID3D12PipelineState* pso = nullptr;
	if (pipelineState != nullptr)
	{
		pso = pipelineState->GetComPtr().Get();
	}

	auto currentCmdAlloc = v_cmdAllocators[m_currentFrame];

	ThrowIfFailed(currentCmdAlloc->GetComPtr()->Reset());
	ThrowIfFailed(m_commandList->Reset(currentCmdAlloc->Get(), pso));
}

void CommandList::ResetAllAllocators()
{
	for (auto& cmdAlloc : v_cmdAllocators)
	{
		ThrowIfFailed(cmdAlloc->GetComPtr()->Reset());
	}
}

void CommandList::TransitionResource(ID3D12Resource* resource, D3D12_RESOURCE_STATES beforeState, D3D12_RESOURCE_STATES afterState)
{
	auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(resource, beforeState, afterState);
	m_commandList->ResourceBarrier(1, &barrier);
}

void CommandList::Close()
{
	ThrowIfFailed(m_commandList->Close());
}

void CommandList::CreateCmdAllocators(Device& device)
{
	assert(m_nFrameCount > 0 && "Frame count must be greater than 0");

	for (int i = 0; i < m_nFrameCount; i++)
	{
		v_cmdAllocators.push_back(new CommandAllocator(device, m_cmdType));
	}
}



CommandList::~CommandList()
{
	for (auto cmdAlloc : v_cmdAllocators) 
	{ 
		delete cmdAlloc; 
	}
};

