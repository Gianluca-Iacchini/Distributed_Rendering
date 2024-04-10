#include "Helpers.h"

#ifndef COMMANDLIST_H
#define COMMANDLIST_H

class Device;
class CommandAllocator;
class PipelineState;

class CommandList
{
public:
	CommandList(Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList);
	CommandList(Device& device, int frameCount = 3, D3D12_COMMAND_LIST_TYPE = D3D12_COMMAND_LIST_TYPE_DIRECT, PipelineState* pipelineState = nullptr);
	

	void Reset(PipelineState* pipelineState = nullptr);
	void ResetAllAllocators();
	void TransitionResource(ID3D12Resource* resource, D3D12_RESOURCE_STATES beforeState, D3D12_RESOURCE_STATES afterState);
	void Close();

	CommandAllocator* GetCurrentCmdAllocator() { return v_cmdAllocators[m_currentFrame]; };

	~CommandList();

	ID3D12GraphicsCommandList* Get() const { return m_commandList.Get(); }
	ID3D12GraphicsCommandList** GetAddressOf() { return m_commandList.GetAddressOf(); }
	Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> GetComPtr() const { return m_commandList; }

private:
	void CreateCmdAllocators(Device& device);

private:
	Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_commandList;
	std::vector<CommandAllocator* > v_cmdAllocators;
	D3D12_COMMAND_LIST_TYPE m_cmdType = D3D12_COMMAND_LIST_TYPE_DIRECT;

	int m_nFrameCount = 3;
	int m_currentFrame = 0;
};

#endif // COMMANDLIST_H



