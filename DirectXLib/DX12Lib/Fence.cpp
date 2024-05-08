#include "pch.h"

#include "Fence.h"

using namespace DX12Lib;

Fence::Fence(Device& device, UINT64 lastFenceValue, UINT64 nextFenceValue)
	: m_lastFenceValue(lastFenceValue), CurrentFenceValue(nextFenceValue)
{
	ThrowIfFailed(device.GetComPtr()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(m_fence.GetAddressOf())));
	m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
}

Fence::Fence(Microsoft::WRL::ComPtr<ID3D12Fence> fence)
	: m_fence(fence)
{
}

Fence::~Fence()
{
	CloseHandle(m_fenceEvent);
}

UINT64 Fence::GetGPUFenceValue()
{
	return m_fence->GetCompletedValue();
}


bool Fence::IsFenceComplete(UINT64 fenceValue)
{
	if (fenceValue > m_lastFenceValue)
		m_lastFenceValue = std::max(m_lastFenceValue, m_fence->GetCompletedValue());

	return fenceValue <= m_lastFenceValue;
}


void Fence::WaitForFence(UINT64 value)
{
	if (!IsFenceComplete(value))
	{
		std::lock_guard<std::mutex> lock(m_fenceEventMutex);

		m_fence->SetEventOnCompletion(value, m_fenceEvent);
		WaitForSingleObject(m_fenceEvent, INFINITE);
		m_lastFenceValue = value;
	}
}

void Fence::WaitForCurrentFence()
{
	WaitForFence(CurrentFenceValue);
}
