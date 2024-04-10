#include "CommandAllocator.h"
#include "Device.h"

using namespace Microsoft::WRL;

CommandAllocator::CommandAllocator(Device& device, D3D12_COMMAND_LIST_TYPE type)
{
	ThrowIfFailed(device.GetComPtr()->CreateCommandAllocator(type, IID_PPV_ARGS(&m_commandAllocator)));
}

CommandAllocator::~CommandAllocator()
{
}