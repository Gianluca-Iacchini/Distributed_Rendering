#include "DX12Lib/Helpers.h"
#include "DX12Lib/CommandAllocator.h"
#include "DX12Lib/Device.h"

#ifndef MAINFRAME_RESOURCE_H
#define MAINFRAME_RESOURCE_H

struct FrameResource
{
public:
	FrameResource(Device& device)
	{
		CmdListAlloc = std::make_shared<CommandAllocator>(device, D3D12_COMMAND_LIST_TYPE_DIRECT);
	}
	FrameResource(const FrameResource& rhs) = delete;
	FrameResource& operator=(const FrameResource& rhs) = delete;
	~FrameResource() {}

	std::shared_ptr<CommandAllocator> CmdListAlloc;

	UINT64 Fence = 0;
};

#endif // !FRAME_RESOURCE_H




