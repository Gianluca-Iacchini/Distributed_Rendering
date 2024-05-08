#include "pch.h"
#include "CIDXGIFactory.h"

using namespace Microsoft::WRL;
using namespace DX12Lib;

CIDXGIFactory::CIDXGIFactory(UINT flags)
{
	ThrowIfFailed(CreateDXGIFactory2(flags, IID_PPV_ARGS(m_factory.GetAddressOf())));
}

CIDXGIFactory::~CIDXGIFactory()
{
}

