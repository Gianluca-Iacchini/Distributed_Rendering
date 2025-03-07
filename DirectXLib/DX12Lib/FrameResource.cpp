#include "pch.h"

#include "FrameResource.h"

using namespace DX12Lib;

FrameResource::FrameResource()
{

}

void FrameResource::InitializeApp(Device& device, D3D12_COMMAND_LIST_TYPE cmdListType)
{
	m_frameCommandAllocator = std::make_unique<CommandAllocator>(device, cmdListType);
}

void FrameResource::EndFrame(CommandQueue& cmdQueue)
{

}

FrameResourceManager::FrameResourceManager(Device& device, UINT nFrameResources) : nFrameResources(nFrameResources)
{
	for (UINT i = 0; i < nFrameResources; i++)
	{
		m_frameResources.push_back(std::make_unique<FrameResource>());
		m_frameResources[i]->InitializeApp(device, D3D12_COMMAND_LIST_TYPE_DIRECT);
	}
}

FrameResource* FrameResourceManager::GetCurrentFrameResource()
{
	assert(m_frameResources[m_currentFrameResourceIndex] != nullptr);
	return m_frameResources[m_currentFrameResourceIndex].get();
}
