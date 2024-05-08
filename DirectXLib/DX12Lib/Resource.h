#pragma once

#include <wrl.h>
#include <d3d12.h>

namespace DX12Lib {

	class Resource
	{
		friend class CommandContext;

	public:
		Resource() :
			m_gpuVirtualAddress(D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN),
			m_currentState(D3D12_RESOURCE_STATE_COMMON),
			m_nextState((D3D12_RESOURCE_STATES)-1)
		{}
		Resource(ID3D12Resource* resource, D3D12_RESOURCE_STATES currentState) :
			m_gpuVirtualAddress(D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN),
			m_resource(resource),
			m_currentState(currentState),
			m_nextState((D3D12_RESOURCE_STATES)-1)
		{}

		~Resource() { OnDestroy(); }

		D3D12_RESOURCE_DESC GetDesc() const { return m_resource->GetDesc(); }

		virtual void OnDestroy()
		{
			m_resource = nullptr;
			m_gpuVirtualAddress = D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN;
			m_versionID += 1;
		}

	protected:

		Microsoft::WRL::ComPtr<ID3D12Resource> m_resource;
		D3D12_GPU_VIRTUAL_ADDRESS m_gpuVirtualAddress;
		D3D12_RESOURCE_STATES m_currentState;
		D3D12_RESOURCE_STATES m_nextState;

		uint32_t m_versionID = 0;

	public:

		ID3D12Resource* operator->() const { return m_resource.Get(); }
		operator ID3D12Resource* () const { return m_resource.Get(); }

		ID3D12Resource* Get() const { return m_resource.Get(); }
		ID3D12Resource** GetAddressOf() { return m_resource.GetAddressOf(); }
		Microsoft::WRL::ComPtr<ID3D12Resource> GetComPtr() const { return m_resource; }
		void ResetComPtr() { m_resource.Reset(); }

		Resource(Resource&&) = default;
		Resource& operator=(Resource&&) = default;

		Resource(Resource&) = delete;
		Resource& operator=(Resource&) = delete;
	};
}


