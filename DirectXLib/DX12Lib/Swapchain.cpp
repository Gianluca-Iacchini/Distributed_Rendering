#include "Swapchain.h"
#include "CIDXGIFactory.h"
#include "DX12Window.h"
#include "Device.h"
#include "CommandQueue.h"
#include "Resource.h"

using namespace Microsoft::WRL;


Swapchain::Swapchain(CIDXGIFactory& factory, CommandQueue& commandQueue, DX12Window& window, int nBufferCount, DXGI_FORMAT backBufferFormat) : 
	m_backBufferFormat(backBufferFormat) ,BufferCount(nBufferCount)
{
	m_swapchain.Reset();

	DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
	swapChainDesc.Width = window.GetWidth(); // Make sure these functions return the actual size
	swapChainDesc.Height = window.GetHeight();
	swapChainDesc.Format = backBufferFormat;
	swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swapChainDesc.BufferCount = BufferCount;
	swapChainDesc.SampleDesc.Count = 1;
	swapChainDesc.SampleDesc.Quality = 0;
	swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	swapChainDesc.Scaling = DXGI_SCALING_STRETCH;
	swapChainDesc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
	swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

	ThrowIfFailed(factory.GetComPointer()->CreateSwapChainForHwnd(
		commandQueue.Get(), // The command queue associated with rendering
		window.GetWindowHandle(),       // Handle to the window
		&swapChainDesc,     // Swap chain description
		nullptr,            // Fullscreen swap chain desc (nullptr for windowed mode)
		nullptr,            // Restrict output to a specific monitor (nullptr for default)
		m_swapchain.GetAddressOf()         // Receives the created swap chain interface
	));

	for (int i = 0; i < BufferCount; i++)
	{
		ComPtr<ID3D12Resource> backBuffer;
		ThrowIfFailed(m_swapchain->GetBuffer(i, IID_PPV_ARGS(backBuffer.GetAddressOf())));
		m_backBufferResources.push_back(std::make_unique<Resource>(Resource(backBuffer)));
	}
}

Swapchain::Swapchain(Microsoft::WRL::ComPtr<IDXGISwapChain1> swapChain) : m_swapchain(swapChain)
{
}

Swapchain::~Swapchain()
{
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
