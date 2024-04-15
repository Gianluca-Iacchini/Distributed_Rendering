#include "Helpers.h"

#ifndef COMMAND_ALLOCATOR_H
#define COMMAND_ALLOCATOR_H

class Device;

class CommandAllocator
{
public:
	CommandAllocator(Device& device, D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT);
	~CommandAllocator();
	void Reset() { ThrowIfFailed(m_commandAllocator->Reset()); }


private:
	Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_commandAllocator;

public:
	ID3D12CommandAllocator* Get() const { return m_commandAllocator.Get(); }
	ID3D12CommandAllocator** GetAddressOf() { return m_commandAllocator.GetAddressOf(); }
	Microsoft::WRL::ComPtr<ID3D12CommandAllocator> GetComPtr() const { return m_commandAllocator; }
};
#endif // COMMAND_ALLOCATOR_H



