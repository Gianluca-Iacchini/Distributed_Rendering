#include "CommandAllocator.h"
#include "Device.h"
#include "GraphicsCore.h"

using namespace Microsoft::WRL;

CommandAllocator::CommandAllocator(Device& device, D3D12_COMMAND_LIST_TYPE type)
{
	ThrowIfFailed(device.GetComPtr()->CreateCommandAllocator(type, IID_PPV_ARGS(&m_commandAllocator)));
}

CommandAllocator::~CommandAllocator()
{
}

CommandAllocatorPool::CommandAllocatorPool(D3D12_COMMAND_LIST_TYPE type)
	: m_type(type)
{
}

CommandAllocatorPool::~CommandAllocatorPool()
{
	m_commandAllocatorPool.clear();
}

std::shared_ptr<CommandAllocator> CommandAllocatorPool::RequestAllocator(uint64_t completedFenceValue)
{
	std::lock_guard<std::mutex> lock(m_cmdAllocatorMutex);

	// If allocator is in the pool then we check if it is ready to be reused, reset it and return it
	if (!m_availableCommandAllocators.empty())
	{
		auto pair = m_availableCommandAllocators.front();

		if (pair.first <= completedFenceValue)
		{
			auto allocator = pair.second;
			allocator->Reset();
			m_availableCommandAllocators.pop();

			return allocator;
		}
	}

	// Otherwise we create a new allocator and return it
	auto cmdAllocator = std::make_shared<CommandAllocator>(*(Graphics::s_device), m_type);
	m_commandAllocatorPool.push_back(cmdAllocator);
	return cmdAllocator;
	
}

void CommandAllocatorPool::DiscardAllocator(uint64_t fenceValue, const std::shared_ptr<CommandAllocator> allocator)
{
	std::lock_guard<std::mutex> lock(m_cmdAllocatorMutex);
	
	m_availableCommandAllocators.push(std::make_pair(fenceValue, allocator));
}
