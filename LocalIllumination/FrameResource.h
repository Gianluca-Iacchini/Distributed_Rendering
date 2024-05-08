#include "DX12Lib/Helpers.h"
#include "DX12Lib/CommandAllocator.h"
#include "DX12Lib/Device.h"
#include "DX12Lib/GraphicsCore.h"

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




