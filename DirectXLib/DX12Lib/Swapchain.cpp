#include "pch.h"
#include "Swapchain.h"
#include "CIDXGIFactory.h"
#include "DX12Window.h"
#include "CommandQueue.h"
#include "Resource.h"

using namespace Microsoft::WRL;
using namespace DX12Lib;


Swapchain::Swapchain(DX12Window& window, DXGI_FORMAT backBufferFormat) : 
	m_window(window), m_backBufferFormat(backBufferFormat)
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
	m_swapchainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH | DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;

}

Swapchain::~Swapchain()
{
}

void Swapchain::Initialize(CommandQueue& commandQueue)
{
	m_swapchain.Reset();

	UINT flags = 0;

#ifdef DEBUG || _DEBUG
		flags = DXGI_CREATE_FACTORY_DEBUG;
#endif // DEBUG

	auto factory = CIDXGIFactory(flags);

	ThrowIfFailed(factory->CreateSwapChainForHwnd(
		commandQueue.Get(), // The command queue associated with rendering
		m_window.GetWindowHandle(),       // Handle to the window
		&m_swapchainDesc,     // Swap chain description
		nullptr,            // Fullscreen swap chain desc (nullptr for windowed mode)
		nullptr,            // Restrict output to a specific monitor (nullptr for default)
		m_swapchain.GetAddressOf()         // Receives the created swap chain interface
	));
}

void Swapchain::Resize(UINT width, UINT height)
{

	for (int i = 0; i < BufferCount;  i++)
	{
		m_backBuffers[i].ResetComPtr();
	}

	ThrowIfFailed(m_swapchain->ResizeBuffers(BufferCount, width, height, 
		m_backBufferFormat, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH | DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING));

	for (int i = 0; i < BufferCount; i++)
	{
		ComPtr<ID3D12Resource> backBuffer;
		ThrowIfFailed(m_swapchain->GetBuffer(i, IID_PPV_ARGS(backBuffer.GetAddressOf())));
		m_backBuffers[i].CreateFromSwapChain(backBuffer);
	}
}
