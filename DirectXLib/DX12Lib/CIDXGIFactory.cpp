#include "CIDXGIFactory.h"
#include "Swapchain.h"

using namespace Microsoft::WRL;

CIDXGIFactory::CIDXGIFactory()
{
	UINT factoryFlags = 0;
#ifdef DEBUG
	factoryFlags = DXGI_CREATE_FACTORY_DEBUG;
#endif // DEBUG

	ThrowIfFailed(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(m_factory.GetAddressOf())));
}

CIDXGIFactory::~CIDXGIFactory()
{
}


//Swapchain CIDXGIFactory::CreateSwapchain(DXGI_SWAP_CHAIN_DESC swapChainDesc)
//{
//	//m_factory->CreateSwapChain()
//}
