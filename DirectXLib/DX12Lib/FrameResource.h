#include "Helpers.h"


#ifndef FRAME_RESOURCE_H
#define FRAME_RESOURCE_H

class CommandAllocator;
class Device;
class CommandQueue;

class FrameResource
{
public:
	FrameResource();
	void Initialize(Device& device, D3D12_COMMAND_LIST_TYPE commandListType = D3D12_COMMAND_LIST_TYPE_DIRECT);
	void EndFrame(CommandQueue& cmdQueue);
	CommandAllocator* GetCommandAllocator() { return m_frameCommandAllocator.get(); }
	~FrameResource() {};

public:
	UINT64 FenceValue = 0;

private:
	std::shared_ptr<CommandAllocator> m_frameCommandAllocator = nullptr;
};

class FrameResourceManager
{
public:
	FrameResourceManager() {};
	FrameResourceManager(Device& device, UINT nFrameResources);
	~FrameResourceManager() {};

	UINT GetFrameResourceIndex() { return m_currentFrameResourceIndex; }
	UINT Increment() { return m_currentFrameResourceIndex = (m_currentFrameResourceIndex + 1) % nFrameResources; }
	FrameResource* GetCurrentFrameResource();
public:
	const UINT nFrameResources = 3;

private:
	UINT m_currentFrameResourceIndex = 0;
	std::vector<std::unique_ptr<FrameResource>> m_frameResources;
};

#endif // !FRAME_RESOURCE_H



