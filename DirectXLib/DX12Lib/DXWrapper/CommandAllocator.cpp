#include "DX12Lib/pch.h"
#include "CommandAllocator.h"

using namespace DX12Lib;

using namespace Microsoft::WRL;

CommandAllocator::CommandAllocator(Device& device, D3D12_COMMAND_LIST_TYPE type, std::wstring debugName)
{
	ThrowIfFailed(device.GetComPtr()->CreateCommandAllocator(type, IID_PPV_ARGS(&m_commandAllocator)));

#ifdef _DEBUG || DEBUG
	m_commandAllocator->SetName(debugName.c_str());
#endif
}

CommandAllocator::~CommandAllocator()
{

}

CommandAllocatorPool::CommandAllocatorPool(D3D12_COMMAND_LIST_TYPE type)
	: m_type(type)
{
	m_availableCommandAllocators = std::queue<std::pair<UINT64, CommandAllocator*>>();

}

CommandAllocatorPool::~CommandAllocatorPool()
{
	for (auto allocator : m_commandAllocatorPool)
	{
		delete allocator;
	}

	m_commandAllocatorPool.clear();
}

CommandAllocator* CommandAllocatorPool::RequestAllocator(UINT64 completedFenceValue)
{
	static UINT m_allocatorCount = 0;

	std::lock_guard<std::mutex> lock(m_cmdAllocatorMutex);

	// If allocator is in the pool then we check if it is ready to be reused, reset it and return it
	if (!m_availableCommandAllocators.empty())
	{

		std::pair<UINT64, CommandAllocator*>& pair = m_availableCommandAllocators.front();

		if (pair.first <= completedFenceValue)
		{

			CommandAllocator* allocator = pair.second;
			allocator->Reset();
			m_availableCommandAllocators.pop();

			std::wstring reusedName = L"ReusedAllocator" + std::to_wstring(m_allocatorCount);
			allocator->Get()->SetName(reusedName.c_str());
			m_allocatorCount++;

			return allocator;
		}
	}

	// Otherwise we create a new allocator and return it
	CommandAllocator* cmdAllocator = new CommandAllocator(*(Graphics::s_device), m_type, L"NewAllocator" + std::to_wstring(m_commandAllocatorPool.size()));
	m_commandAllocatorPool.push_back(cmdAllocator);

	return cmdAllocator;
	
}

void CommandAllocatorPool::DiscardAllocator(UINT64 fenceValue, CommandAllocator* allocator)
{
	std::lock_guard<std::mutex> lock(m_cmdAllocatorMutex);
	
	m_availableCommandAllocators.push(std::make_pair(fenceValue, allocator));
}
