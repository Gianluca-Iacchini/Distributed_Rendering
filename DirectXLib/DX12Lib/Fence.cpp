#include "Fence.h"
#include "Device.h"

Fence::Fence(Device& device, UINT64 initialValue)
{
	ThrowIfFailed(device.GetComPtr()->CreateFence(initialValue, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(m_fence.GetAddressOf())));
}

Fence::Fence(Microsoft::WRL::ComPtr<ID3D12Fence> fence)
	: m_fence(fence)
{
}

Fence::~Fence()
{
}

UINT64 Fence::GetCPUFenceValue()
{
	return 0;
}

UINT64 Fence::GetGPUFenceValue()
{
	return 0;
}


bool Fence::IsFenceComplete(UINT64 fenceValue)
{
	return false;
}

void Fence::WaitForFence()
{
	if (m_fence->GetCompletedValue() < m_fenceValue)
	{
		m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
		ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValue, m_fenceEvent));

		WaitForSingleObject(m_fenceEvent, INFINITE);
		CloseHandle(m_fenceEvent);
	}
}

void Fence::SetFenceValue(UINT64 fenceValue)
{
}

void Fence::IncreaseCounter()
{
}
