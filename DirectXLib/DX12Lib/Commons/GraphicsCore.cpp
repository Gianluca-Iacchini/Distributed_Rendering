#include "DX12Lib/pch.h"
#include "GraphicsCore.h"
#include "DX12Lib/DXWrapper/CIDXGIFactory.h"
#include "DX12Lib/DXWrapper/Adapter.h"
#include "dxgidebug.h"

#define DRED 0


using namespace Microsoft::WRL;
using namespace DX12Lib;


namespace Graphics
{
	DescriptorAllocator s_descriptorAllocators[D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES]
	{
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
		D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
		D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
		D3D12_DESCRIPTOR_HEAP_TYPE_DSV
	};


	DXGI_FORMAT m_backBufferFormat = DXGI_FORMAT_B8G8R8A8_UNORM;
	
	DXGI_FORMAT m_depthStencilFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;

	std::shared_ptr<Device> Graphics::s_device = nullptr;
	std::unique_ptr<CommandQueueManager> Graphics::s_commandQueueManager = nullptr;
	std::unique_ptr<CommandContextManager> s_commandContextManager = nullptr;
	std::unique_ptr<DirectX::Mouse> s_mouse = nullptr;
	std::unique_ptr<DirectX::Keyboard> s_keyboard = nullptr;
	std::unique_ptr<DirectX::Keyboard::KeyboardStateTracker> s_kbTracker = nullptr;
	std::unique_ptr<DirectX::Mouse::ButtonStateTracker> s_mouseTracker = nullptr;
	Microsoft::WRL::ComPtr<ID3D12DeviceRemovedExtendedDataSettings1> s_dredSettings = nullptr;
	std::shared_ptr<DX12Lib::QueryHeap> s_queryHeap = nullptr;
	UINT64 s_gpuGraphicsFrequency = 1;
	UINT64 s_gpuComputeFrequency = 1;

	void LogAdapterOutput(ComPtr<IDXGIAdapter> adapter)
	{
		UINT i = 0;
		ComPtr<IDXGIOutput> output = nullptr;
		while (adapter->EnumOutputs(i, output.GetAddressOf()) != DXGI_ERROR_NOT_FOUND)
		{
			DXGI_OUTPUT_DESC desc;
			output->GetDesc(&desc);

			std::wstring text = L"***Output: ";
			text += desc.DeviceName;
			text += L"\n";
			OutputDebugStringW(text.c_str());

			if (output)
				output->Release();

			i++;
		}
	}

	void LogAdapters(CIDXGIFactory& factory)
	{
		UINT i = 0;
		ComPtr<IDXGIAdapter> adapter = nullptr;
		std::vector<ComPtr<IDXGIAdapter>> adapterList;

		while (factory->EnumAdapters(i, adapter.GetAddressOf()) != DXGI_ERROR_NOT_FOUND)
		{
			DXGI_ADAPTER_DESC desc;
			adapter->GetDesc(&desc);

			std::wstring text = L"***Adapter: ";
			text += desc.Description;
			text += L"\n";

			OutputDebugStringW(text.c_str());

			adapterList.push_back(adapter);
			i++;
		}

		for (size_t i = 0; i < adapterList.size(); i++)
		{
			LogAdapterOutput(adapterList[i]);

			if (adapterList[i])
				adapterList[i]->Release();
		}
	}

	bool InitializeApp()
	{
		UINT dxgiFactoryFlags = 0;

		#if defined(DEBUG) || defined (_DEBUG)
		{
			Microsoft::WRL::ComPtr<ID3D12Debug> debugController;
			ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)));
			debugController->EnableDebugLayer();
			dxgiFactoryFlags = DXGI_CREATE_FACTORY_DEBUG;
		}
		#endif

		#if DRED
		{
			ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(&s_dredSettings)));
			s_dredSettings->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
			s_dredSettings->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
		}
		#endif
		auto factory = CIDXGIFactory(dxgiFactoryFlags);

		s_device = std::make_shared<Device>();

		if (!s_device->InitializeApp(nullptr))
		{
			Adapter warpAdapter = Adapter(factory, true);

			if (!s_device->InitializeApp(&warpAdapter))
			{
				MessageBox(0, L"Direct3D initialization failed.", 0, 0);
				return false;
			}
		}

		#ifdef _DEBUG
			LogAdapters(factory);
			ID3D12DebugDevice* debugDevice = nullptr;
			if (SUCCEEDED(s_device->GetComPtr()->QueryInterface(&debugDevice)))
			{
				debugDevice->ReportLiveDeviceObjects(D3D12_RLDO_DETAIL | D3D12_RLDO_IGNORE_INTERNAL);
				debugDevice->Release();
			}
		#endif // _DEBUG


			s_commandQueueManager = std::make_unique<CommandQueueManager>(*s_device);
			s_commandQueueManager->Create();
			s_commandContextManager = std::make_unique<CommandContextManager>();
			s_keyboard = std::make_unique<DirectX::Keyboard>();
			s_kbTracker = std::make_unique<DirectX::Keyboard::KeyboardStateTracker>();
			s_mouseTracker = std::make_unique<DirectX::Mouse::ButtonStateTracker>();

			s_queryHeap = std::make_shared<QueryHeap>();
			s_queryHeap->Create(D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 64);

			ThrowIfFailed(s_commandQueueManager->GetGraphicsQueue().Get()->GetTimestampFrequency(&s_gpuGraphicsFrequency));
			ThrowIfFailed(s_commandQueueManager->GetComputeQueue().Get()->GetTimestampFrequency(&s_gpuComputeFrequency));

			Renderer::InitializeApp();

		return true;
	}

	void Shutdown()
	{
		s_commandQueueManager->GetGraphicsQueue().Flush();
		s_commandQueueManager->GetComputeQueue().Flush();
		s_commandQueueManager->GetCopyQueue().Flush();

		if (s_device != nullptr)
		{
			auto hr = s_device->GetComPtr()->GetDeviceRemovedReason();
			ThrowIfFailed(hr);
		}
		
		Renderer::Shutdown();

		s_device = nullptr;
		s_commandQueueManager = nullptr;
		s_commandContextManager = nullptr;
		s_mouse = nullptr;
	}

	void DeviceRemovedHandler()
	{
#if DRED
		Microsoft::WRL::ComPtr<ID3D12DeviceRemovedExtendedData1> pDred;
		ThrowIfFailed(s_device->GetComPtr()->QueryInterface(IID_PPV_ARGS(&pDred)));

		D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 DredAutoBreadcrumbsOutput;
		D3D12_DRED_PAGE_FAULT_OUTPUT DredPageFaultOutput;
		ThrowIfFailed(pDred->GetAutoBreadcrumbsOutput1(&DredAutoBreadcrumbsOutput));
		ThrowIfFailed(pDred->GetPageFaultAllocationOutput(&DredPageFaultOutput));
		

		auto crumb = DredAutoBreadcrumbsOutput.pHeadAutoBreadcrumbNode;

		while (crumb != nullptr)
		{
			for (UINT i = 0; i < crumb->BreadcrumbCount; i++)
			{
				std::cout << "***DRED Auto Breadcrumb: " << crumb->pCommandHistory[i] << std::endl;
				std::cout << "LAST VALUE: " << crumb->pLastBreadcrumbValue << std::endl;
			} 
			crumb = crumb->pNext;
		}
#endif
	}

	UINT64 GetGraphicsGPUFrequency()
	{
		return s_gpuGraphicsFrequency;
	}

	UINT64 GetComputeGPUFrequency()
	{
		return s_gpuComputeFrequency;
	}
}

