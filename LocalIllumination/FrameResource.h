#include "DX12Lib/Commons/Helpers.h"
#include "DX12Lib/DXWrapper/CommandAllocator.h"
#include "DX12Lib/DXWrapper/Device.h"
#include "DX12Lib/Commons/GraphicsCore.h"

#ifndef MAINFRAME_RESOURCE_H
#define MAINFRAME_RESOURCE_H

struct FrameResource
{
public:
	FrameResource(DX12Lib::Device& device, uint64_t value)
	{

	}
	FrameResource(const FrameResource& rhs) = delete;
	FrameResource& operator=(const FrameResource& rhs) = delete;
	~FrameResource() 
	{

	}

	DX12Lib::CommandAllocator* CmdListAlloc;

	UINT64 Fence = 0;
};

#endif // !FRAME_RESOURCE_H




