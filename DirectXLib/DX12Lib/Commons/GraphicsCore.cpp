#include "DX12Lib/pch.h"
#include "GraphicsCore.h"
#include "DX12Lib/DXWrapper/CIDXGIFactory.h"
#include "DX12Lib/DXWrapper/Adapter.h"
#include "dxgidebug.h"



using namespace Microsoft::WRL;
using namespace DX12Lib;

#define DRED

namespace Graphics
{
	DescriptorAllocator s_descriptorAllocators[D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES]
	{
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
		D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
		D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
		D3D12_DESCRIPTOR_HEAP_TYPE_DSV
	};

	DXGI_FORMAT m_backBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
	DXGI_FORMAT m_depthStencilFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;

	std::shared_ptr<Device> Graphics::s_device = nullptr;

	std::unique_ptr<CommandQueueManager> Graphics::s_commandQueueManager = nullptr;

	std::unique_ptr<CommandContextManager> s_commandContextManager = nullptr;

	std::unique_ptr<DirectX::GraphicsMemory> s_graphicsMemory = nullptr;

	Microsoft::WRL::ComPtr<ID3D12DeviceRemovedExtendedDataSettings1> s_dredSettings = nullptr;

	std::unique_ptr<DirectX::Mouse> s_mouse = nullptr;

	std::unique_ptr<TextureManager> s_textureManager = nullptr;

	std::unique_ptr<MaterialManager> s_materialManager = nullptr;

	std::shared_ptr<DX12Lib::DescriptorHeap> s_textureHeap = nullptr;

	std::unordered_map<std::wstring, std::shared_ptr<PipelineState>> s_PSOs;

	std::unordered_map<std::wstring, std::shared_ptr<Shader>> s_shaders;

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

	void CreateDefaultShaders()
	{
		std::wstring srcDir = Utils::ToWstring(SOURCE_DIR);
		std::wstring VSshaderFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\Basic_VS.hlsl";
		std::wstring PSshaderFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\Basic_PS.hlsl";
		std::wstring PBRPSShaderFile = srcDir + L"\\DX12Lib\\DXWrapper\\Shaders\\BasicPBR_PS.hlsl";

		std::shared_ptr<Shader> baseVertexShader = std::make_shared<Shader>(VSshaderFile, "VS", "vs_5_1");
		std::shared_ptr<Shader> basePixelShader = std::make_shared<Shader>(PSshaderFile, "PS", "ps_5_1");
		std::shared_ptr<Shader> PBRPixelShader = std::make_shared<Shader>(PBRPSShaderFile, "PS", "ps_5_1");

		baseVertexShader->Compile();
		basePixelShader->Compile();
		PBRPixelShader->Compile();

		s_shaders[L"basicVS"] = std::move(baseVertexShader);
		s_shaders[L"basicPS"] = std::move(basePixelShader);
		s_shaders[L"PBRBasicPS"] = std::move(PBRPixelShader);
	}

	void CreateDefaultPSOs()
	{
		SamplerDesc DefaultSamplerDesc;
		DefaultSamplerDesc.MaxAnisotropy = 8;

		std::shared_ptr<RootSignature> baseRootSignature = std::make_shared<RootSignature>(5, 1);
		baseRootSignature->InitStaticSampler(0, DefaultSamplerDesc);
		(*baseRootSignature)[0].InitAsConstantBuffer(0);
		(*baseRootSignature)[1].InitAsConstantBuffer(1);
		(*baseRootSignature)[2].InitAsConstantBuffer(2);
		(*baseRootSignature)[3].InitAsBufferSRV(3, D3D12_SHADER_VISIBILITY_PIXEL, 1);
		(*baseRootSignature)[4].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, NUM_PHONG_TEXTURES);
		baseRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

		std::shared_ptr<RootSignature> pbrRootSignature = std::make_shared<RootSignature>(5, 1);
		pbrRootSignature->InitStaticSampler(0, DefaultSamplerDesc);
		(*pbrRootSignature)[0].InitAsConstantBuffer(0);
		(*pbrRootSignature)[1].InitAsConstantBuffer(1);
		(*pbrRootSignature)[2].InitAsConstantBuffer(2);
		(*pbrRootSignature)[3].InitAsBufferSRV(3, D3D12_SHADER_VISIBILITY_PIXEL, 1);
		(*pbrRootSignature)[4].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, NUM_PBR_TEXTURES);
		pbrRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);





		std::shared_ptr<PipelineState> opaquePSO = std::make_shared<PipelineState>();
		opaquePSO->InitializeDefaultStates();
		opaquePSO->SetInputLayout(DirectX::VertexPositionNormalTexture::InputLayout.pInputElementDescs,\
			DirectX::VertexPositionNormalTexture::InputLayout.NumElements);
		opaquePSO->SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
		opaquePSO->SetRenderTargetFormat(m_backBufferFormat, m_depthStencilFormat, 1, 0);
		opaquePSO->SetShader(s_shaders[L"basicVS"], ShaderType::Vertex);
		opaquePSO->SetShader(s_shaders[L"basicPS"], ShaderType::Pixel);
		opaquePSO->SetRootSignature(baseRootSignature);
		opaquePSO->Finalize();

		// Duplicate content of opaquePSO
		std::shared_ptr<PipelineState> PBRPSO = std::make_shared<PipelineState>();
		PBRPSO->InitializeDefaultStates();
		PBRPSO->SetInputLayout(DirectX::VertexPositionNormalTexture::InputLayout.pInputElementDescs, \
			DirectX::VertexPositionNormalTexture::InputLayout.NumElements);
		PBRPSO->SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
		PBRPSO->SetRenderTargetFormat(m_backBufferFormat, m_depthStencilFormat, 1, 0);
		PBRPSO->SetShader(s_shaders[L"basicVS"], ShaderType::Vertex);
		PBRPSO->SetShader(s_shaders[L"PBRBasicPS"], ShaderType::Pixel);
		PBRPSO->SetRootSignature(pbrRootSignature);
		PBRPSO->Finalize();





		s_PSOs[L"opaquePSO"] = std::move(opaquePSO);
		s_PSOs[L"PBRPSO"] = std::move(PBRPSO);
	}

	bool Initialize()
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

		#if defined(DRED)
		{
			ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(&s_dredSettings)));
			s_dredSettings->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
			s_dredSettings->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
		}
		#endif
		auto factory = CIDXGIFactory(dxgiFactoryFlags);

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
			s_commandContextManager = std::make_unique<CommandContextManager>();
			s_textureHeap = std::make_shared<DescriptorHeap>();
			s_textureHeap->Create(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4096);
			s_graphicsMemory = std::make_unique<DirectX::GraphicsMemory>(*s_device);
			s_textureManager = std::make_unique<TextureManager>();
			s_materialManager = std::make_unique<MaterialManager>();

			CreateDefaultShaders();
			CreateDefaultPSOs();

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
			s_device = nullptr;
		}

		s_commandQueueManager = nullptr;
		s_commandContextManager = nullptr;
		s_graphicsMemory = nullptr;
	}

	void DeviceRemovedHandler()
	{
#if defined(DRED)
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

}

