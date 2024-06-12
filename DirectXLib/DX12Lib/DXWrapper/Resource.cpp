#include "DX12Lib/pch.h"
#include "Resource.h"


DX12Lib::Resource::Resource() :
	m_gpuVirtualAddress(D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN),
	m_currentState(D3D12_RESOURCE_STATE_COMMON),
	m_nextState((D3D12_RESOURCE_STATES)-1)
{}

DX12Lib::Resource::Resource(ID3D12Resource* resource, D3D12_RESOURCE_STATES currentState) :
	m_gpuVirtualAddress(D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN),
	m_resource(resource),
	m_currentState(currentState),
	m_nextState((D3D12_RESOURCE_STATES)-1)
{}

void DX12Lib::Resource::OnDestroy()
{
	m_resource = nullptr;
	m_gpuVirtualAddress = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN;
	m_versionID += 1;
}
