#include "Swapchain.h"
#include "CIDXGIFactory.h"
#include "DX12Window.h"
#include "Device.h"
#include "CommandQueue.h"
#include "Resource.h"

using namespace Microsoft::WRL;


Swapchain::Swapchain(DX12Window& window, int nBufferCount, DXGI_FORMAT backBufferFormat) : 
	m_backBufferFormat(backBufferFormat) ,BufferCount(nBufferCount)
{
	m_swapchain.Reset();

	m_swapchainDesc.Width = window.GetWidth(); // Make sure these functions return the actual size
	m_swapchainDesc.Height = window.GetHeight();
	m_swapchainDesc.Format = backBufferFormat;
	m_swapchainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	m_swapchainDesc.BufferCount = BufferCount;
	m_swapchainDesc.SampleDesc.Count = 1;
	m_swapchainDesc.SampleDesc.Quality = 0;
	m_swapchainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	m_swapchainDesc.Scaling = DXGI_SCALING_STRETCH;
	m_swapchainDesc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
	m_swapchainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

	m_windowHandle = window.GetWindowHandle();


}

Swapchain::Swapchain(Microsoft::WRL::ComPtr<IDXGISwapChain1> swapChain) : m_swapchain(swapChain)
{
	swapChain->GetDesc1(&m_swapchainDesc);
}

Swapchain::~Swapchain()
{
}

Resource* Swapchain::GetBuffer(UINT index)
{
	assert(index < m_backBufferResources.size() && "Buffer index out of bounds.");

	return m_backBufferResources[index].get();
}

Resource* Swapchain::GetCurrentBackBuffer()
{
	return GetBuffer(m_currentBackBufferIndex);
}



void Swapchain::Resize(UINT width, UINT height)
{
	for (int i = 0; i < m_backBufferResources.size();  i++)
	{
		m_backBufferResources[i]->ResetComPtr();
	}

	ThrowIfFailed(m_swapchain->ResizeBuffers(BufferCount, width, height, 
		m_backBufferFormat, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH));

	for (int i = 0; i < m_backBufferResources.size(); i++)
	{
		ThrowIfFailed(m_swapchain->GetBuffer(i, IID_PPV_ARGS(m_backBufferResources[i]->GetAddressOf())));
	}



}

void Swapchain::Finalize(std::shared_ptr<Device> device, CIDXGIFactory& factory, CommandQueue& commandQueue)
{
	ThrowIfFailed(factory.GetComPointer()->CreateSwapChainForHwnd(commandQueue.Get(),
		m_windowHandle,
		&m_swapchainDesc,
		nullptr,
		nullptr,
		m_swapchain.GetAddressOf()));

	for (int i = 0; i < BufferCount; i++)
	{
		ComPtr<ID3D12Resource> backBuffer;
		ThrowIfFailed(m_swapchain->GetBuffer(i, IID_PPV_ARGS(backBuffer.GetAddressOf())));
		m_backBufferResources.push_back(std::make_unique<Resource>(device, backBuffer));
	}
}