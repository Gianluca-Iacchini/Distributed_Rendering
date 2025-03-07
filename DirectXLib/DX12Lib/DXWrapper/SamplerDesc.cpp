#include "DX12Lib/pch.h"
#include "SamplerDesc.h"


using namespace Graphics;
using namespace DX12Lib;


D3D12_CPU_DESCRIPTOR_HANDLE SamplerDesc::CreateDescriptor(void)
{
    D3D12_CPU_DESCRIPTOR_HANDLE Handle = AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    s_device->GetComPtr()->CreateSampler(this, Handle);
    return Handle;
}

void SamplerDesc::CreateDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE Handle)
{
	s_device->GetComPtr()->CreateSampler(this, Handle);
}
