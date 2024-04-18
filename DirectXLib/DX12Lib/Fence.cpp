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

UINT64 Fence::GetGPUFenceValue()
{
	return m_fence->GetCompletedValue();
}


bool Fence::IsFenceComplete(UINT64 fenceValue)
{
	return false;
}

void Fence::WaitForFence()
{
	WaitForFence(FenceValue);
}

void Fence::WaitForFence(UINT64 value)
{
	if (m_fence->GetCompletedValue() < value)
	{
		HANDLE eventHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
		ThrowIfFailed(m_fence->SetEventOnCompletion(FenceValue, eventHandle));

		WaitForSingleObject(eventHandle, INFINITE);
		CloseHandle(eventHandle);
	}
}
