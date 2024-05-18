#include "DX12Lib/pch.h"
#include "CommandList.h"

using namespace Microsoft::WRL;
using namespace DX12Lib;

CommandList::CommandList(Device& device, CommandAllocator& cmdAllocator, D3D12_COMMAND_LIST_TYPE cmdType, std::shared_ptr<PipelineState> pipelineState)
	: m_cmdType(cmdType), m_pipelineState(pipelineState)
{
	ID3D12PipelineState* pso = nullptr;
	if (pipelineState != nullptr)
	{
		pso = pipelineState->GetComPtr().Get();
	}

	ThrowIfFailed(device.GetComPtr()->CreateCommandList(0, cmdType, cmdAllocator.Get(), pso, IID_PPV_ARGS(&m_commandList)));
}

void DX12Lib::CommandList::SetPipelineState(std::shared_ptr<PipelineState> pipelineState)
{
	m_pipelineState = pipelineState;
	m_commandList->SetPipelineState(m_pipelineState->Get());
}

/// <summary>
/// Reset the command list and the associated command allocator
/// </summary>
/// <param name="cmdAllocator"></param>
void CommandList::Reset(CommandAllocator& cmdAllocator)
{
	ID3D12PipelineState* pso = nullptr;
	if (m_pipelineState != nullptr)
	{
		pso = m_pipelineState->GetComPtr().Get();
	}

	m_closed = false;
	ThrowIfFailed(m_commandList->Reset(cmdAllocator.Get(), pso));
}

void CommandList::TransitionResource(ID3D12Resource* resource, D3D12_RESOURCE_STATES beforeState, D3D12_RESOURCE_STATES afterState)
{
	auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(resource, beforeState, afterState);
	m_commandList->ResourceBarrier(1, &barrier);
}

void CommandList::Close()
{
	if (!m_closed)
		ThrowIfFailed(m_commandList->Close());
	
	m_closed = true;
}



CommandList::~CommandList()
{
};

