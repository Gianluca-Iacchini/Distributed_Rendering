#include "Helpers.h"

#ifndef FENCE_H
#define FENCE_H

class Device;

class Fence
{
public:
	Fence(Device& device, UINT64 initialValue = 0);
	Fence(Microsoft::WRL::ComPtr<ID3D12Fence> fence);
	Fence(const Fence& rhs) = delete;
	Fence& operator=(const Fence& rhs) = delete;
	~Fence();

	UINT64 GetGPUFenceValue();
	bool IsFenceComplete(UINT64 fenceValue);
	void WaitForFence();
	void WaitForFence(UINT64 fenceValue);

	Microsoft::WRL::ComPtr<ID3D12Fence> GetComPtr() const { return m_fence; }
	ID3D12Fence* Get() const { return m_fence.Get(); }
	ID3D12Fence** GetAddressOf() { return m_fence.GetAddressOf(); }

public:
	UINT64 FenceValue = 0;

private:

	Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
	HANDLE m_fenceEvent;
};
#endif // !FENCE_H



