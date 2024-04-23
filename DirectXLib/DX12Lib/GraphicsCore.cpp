#include "GraphicsCore.h"
#include "Device.h"
#include "CIDXGIFactory.h"
#include "Adapter.h"
#include "CommandAllocator.h"
#include "CommandQueue.h"
#include "CommandList.h"
#include "dxgidebug.h"

using namespace Microsoft::WRL;

namespace Graphics
{
	DescriptorAllocator s_descriptorAllocators[D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES]
	{
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
		D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
		D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
		D3D12_DESCRIPTOR_HEAP_TYPE_DSV
	};

	std::shared_ptr<Device> Graphics::s_device = nullptr;

	std::unique_ptr<CommandQueueManager> Graphics::s_commandQueueManager = nullptr;

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

	bool Initialize()
	{
		#if defined(DEBUG) || defined (_DEBUG)
		{
			Microsoft::WRL::ComPtr<ID3D12Debug> debugController;
			ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)));
			debugController->EnableDebugLayer();
		}
		#endif

		auto factory = CIDXGIFactory();

		s_device = std::make_shared<Device>();

		if (!s_device->Initialize(nullptr))
		{
			Adapter warpAdapter = Adapter(factory, true);

			if (!s_device->Initialize(&warpAdapter))
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
		return true;
	}

	void Shutdown()
	{

	}
}

