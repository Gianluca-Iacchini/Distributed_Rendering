#include "Helpers.h"

#ifndef COMMANDLIST_H
#define COMMANDLIST_H

class Device;
class CommandAllocator;
class PipelineState;

class CommandList
{
public:
	CommandList(Device& device, CommandAllocator& commandAllocator, D3D12_COMMAND_LIST_TYPE = D3D12_COMMAND_LIST_TYPE_DIRECT, std::shared_ptr<PipelineState> pipelineState = nullptr);
	
	void SetPipelineState(std::shared_ptr<PipelineState> pipelineState) { m_pipelineState = pipelineState; }
	
	void Reset(CommandAllocator& cmdAllocator);

	void TransitionResource(ID3D12Resource* resource, D3D12_RESOURCE_STATES beforeState, D3D12_RESOURCE_STATES afterState);
	void Close();

	~CommandList();


private:
	Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_commandList;
	std::shared_ptr<PipelineState> m_pipelineState;
	bool m_closed = false;

public:
	const D3D12_COMMAND_LIST_TYPE m_cmdType;
public:
	ID3D12GraphicsCommandList* Get() const { return m_commandList.Get(); }
	ID3D12GraphicsCommandList** GetAddressOf() { return m_commandList.GetAddressOf(); }
	Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> GetComPtr() const { return m_commandList; }

};

#endif // COMMANDLIST_H



