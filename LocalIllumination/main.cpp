#include "DX12Lib/Helpers.h"
#include "DX12Lib/D3DApp.h"
#include <iostream>
#include "DX12Lib/CIDXGIFactory.h"
#include "Keyboard.h"
#include "DX12Lib/Shader.h"
#include "Dx12Lib/PipelineState.h"
#include "DX12Lib/Helpers.h"
#include "DX12Lib/FrameResource.h"

using namespace DirectX;
using namespace Microsoft::WRL;

struct Vertex
{
	Vertex(XMFLOAT3 p, XMFLOAT4 c) : Pos(p), Color(c) {}
	XMFLOAT3 Pos;
	XMFLOAT4 Color;
};

struct VertexResourceData
{
	ComPtr<ID3D12Resource> VertexBufferGPU = nullptr;
	ComPtr<ID3D12Resource> IndexBufferGPU = nullptr;

	ComPtr<ID3DBlob> VertexBufferCPU = nullptr;
	ComPtr<ID3DBlob> IndexBufferCPU = nullptr;

	ComPtr<ID3D12Resource> VertexBufferUploader = nullptr;
	ComPtr<ID3D12Resource> IndexBufferUploader = nullptr;

	UINT VertexBufferStride = 0;
	UINT VertexBufferByteSize = 0;
	DXGI_FORMAT IndexBufferFormat = DXGI_FORMAT_R16_UINT;
	UINT IndexBufferByteSize = 0;

	D3D12_VERTEX_BUFFER_VIEW VertexBufferView() const
	{
		D3D12_VERTEX_BUFFER_VIEW vbv;
		vbv.BufferLocation = VertexBufferGPU->GetGPUVirtualAddress();
		vbv.StrideInBytes = VertexBufferStride;
		vbv.SizeInBytes = VertexBufferByteSize;

		return vbv;
	}

	D3D12_INDEX_BUFFER_VIEW IndexBufferView() const
	{
		D3D12_INDEX_BUFFER_VIEW ibv;
		ibv.BufferLocation = IndexBufferGPU->GetGPUVirtualAddress();
		ibv.Format = IndexBufferFormat;
		ibv.SizeInBytes = IndexBufferByteSize;

		return ibv;
	}
};

class AppTest : public D3DApp
{
	Vertex triangleVertices[3] =
	{
		{ XMFLOAT3(0.0f, 0.5f, 0.0f), XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f) },
		{ XMFLOAT3(0.5f, -0.5f, 0.0f), XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f) },
		{ XMFLOAT3(-0.5f, -0.5f, 0.0f), XMFLOAT4(0.0f, 0.0f, 1.0f, 1.0f) }
	};

	VertexResourceData m_vertexData;

	std::unordered_map<std::string, std::shared_ptr<Shader>> mp_shaders;
	std::unique_ptr<PipelineState> m_pipelineState;

	Keyboard keyboard;
	DirectX::Keyboard::KeyboardStateTracker tracker;

	ComPtr<ID3D12RootSignature> m_rootSignature;

	std::unique_ptr<FrameResourceManager> m_frameResourceManager;

public:
	AppTest(HINSTANCE hInstance) : D3DApp(hInstance) {};
	AppTest(const AppTest& rhs) = delete;
	AppTest& operator=(const AppTest& rhs) = delete;
	~AppTest() { 
		if (m_d3dDevice != nullptr)
			m_commandQueue->Flush();
		
		FreeConsole();
	};

	void BuildRootSignature()
	{
		D3D12_ROOT_SIGNATURE_DESC rootSigDesc = {};
		rootSigDesc.NumParameters = 0;
		rootSigDesc.pParameters = nullptr;
		rootSigDesc.NumStaticSamplers = 0;
		rootSigDesc.pStaticSamplers = nullptr;
		rootSigDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

		ComPtr<ID3DBlob> serializedRootSig = nullptr;
		ComPtr<ID3DBlob> errorBlob = nullptr;
		HRESULT hr = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1, serializedRootSig.GetAddressOf(), errorBlob.GetAddressOf());
		
		if (FAILED(hr))
		{
			if (errorBlob != nullptr)
				OutputDebugStringA((char*)errorBlob->GetBufferPointer());
		}


		ThrowIfFailed(m_d3dDevice->GetComPtr()->CreateRootSignature(0, serializedRootSig->GetBufferPointer(), serializedRootSig->GetBufferSize(), IID_PPV_ARGS(&m_rootSignature)));
		
	}

	void BuildVertexData()
	{
		std::uint16_t triangleIndices[] = { 0, 1, 2 };

		ThrowIfFailed(D3DCreateBlob(sizeof(triangleVertices), &m_vertexData.VertexBufferCPU));
		CopyMemory(m_vertexData.VertexBufferCPU->GetBufferPointer(), triangleVertices, sizeof(triangleVertices));

		ThrowIfFailed(D3DCreateBlob(sizeof(triangleIndices), &m_vertexData.IndexBufferCPU));
		CopyMemory(m_vertexData.IndexBufferCPU->GetBufferPointer(), triangleIndices, sizeof(triangleIndices));

		m_vertexData.VertexBufferGPU = Utils::CreateDefaultBuffer(m_d3dDevice->GetComPtr(), m_commandList->GetComPtr(),
			triangleVertices, sizeof(triangleVertices), m_vertexData.VertexBufferUploader);
		
		m_vertexData.IndexBufferGPU = Utils::CreateDefaultBuffer(m_d3dDevice->GetComPtr(), m_commandList->GetComPtr(),
			triangleIndices, sizeof(triangleIndices), m_vertexData.IndexBufferUploader);

		m_vertexData.VertexBufferByteSize = sizeof(triangleVertices);
		m_vertexData.VertexBufferStride = sizeof(Vertex);
		m_vertexData.IndexBufferByteSize = sizeof(triangleIndices);
		m_vertexData.IndexBufferFormat = DXGI_FORMAT_R16_UINT;
	}

	virtual bool Initialize() override
	{
		if (!D3DApp::Initialize())
			return false;

		m_commandList->Reset(*m_commandAllocator);

		std::cout << "Hello World!" << std::endl;

		std::wstring srcDir = ToWstring(SOURCE_DIR);


		std::wstring shaderPath = srcDir + L"/Shaders/Basic.hlsl";

		std::wcout << shaderPath << std::endl;

		m_frameResourceManager = std::make_unique<FrameResourceManager>(*m_d3dDevice, 3);

		mp_shaders["BasicVS"] = std::make_shared<Shader>(shaderPath, "VS", "vs_5_1");
		mp_shaders["BasicVS"]->InputLayout =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
		};

		mp_shaders["BasicPS"] = std::make_shared<Shader>(shaderPath, "PS", "ps_5_1");

		mp_shaders["BasicVS"]->Compile();
		mp_shaders["BasicPS"]->Compile();

		BuildRootSignature();
		BuildVertexData();

		m_pipelineState = std::make_unique<PipelineState>(mBackBufferFormat, mDepthStencilFormat);
		m_pipelineState->SetShader(mp_shaders["BasicVS"], ShaderType::Vertex);
		m_pipelineState->SetShader(mp_shaders["BasicPS"], ShaderType::Pixel);
		m_pipelineState->SetRootSignature(m_rootSignature.Get());
		m_pipelineState->Finalize(*m_d3dDevice);

		m_commandQueue->ExecuteCommandList(*m_commandList);
		m_commandQueue->Flush();



		return true;
	}

private:

	virtual void OnResize() override
	{
		D3DApp::OnResize();
	}

	virtual void Update(const GameTime& gt) override
	{
		auto kbState = keyboard.GetState();
		tracker.Update(kbState);

		if (tracker.IsKeyPressed(DirectX::Keyboard::Keys::Escape))
		{
			PostQuitMessage(0);
		} 


		m_frameResourceManager->Increment();
		auto currentFrame = m_frameResourceManager->GetCurrentFrameResource();

		m_commandQueue->WaitForFenceValue(currentFrame->FenceValue);
	}

	virtual void Draw(const GameTime& gt) override
	{
		auto currentFrameResource = m_frameResourceManager->GetCurrentFrameResource();

		auto cmdAllocator = currentFrameResource->GetCommandAllocator();
		
		cmdAllocator->Reset();
		m_commandList->Reset(*cmdAllocator);

		m_commandList->GetComPtr()->SetGraphicsRootSignature(m_rootSignature.Get());

		m_commandList->GetComPtr()->RSSetViewports(1, &mScreenViewport);
		m_commandList->GetComPtr()->RSSetScissorRects(1, &mScissorRect);

		auto backBuffer = m_swapchain->GetCurrentBackBuffer();
		auto backBufferRtv = backBuffer->GetView(DescriptorType::RTV);
		auto handle = m_rtvHeap->GetCPUDescriptorHandle(backBufferRtv.DescriptorIndex);

		auto depthStencilView = m_depthStencilBuffer->GetView(DescriptorType::DSV);
		auto depthHandle = m_dsvHeap->GetCPUDescriptorHandle(depthStencilView.DescriptorIndex);



		m_commandList->TransitionResource(m_swapchain->GetCurrentBackBuffer()->Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);


		m_commandList->GetComPtr()->ClearRenderTargetView(handle, Colors::LightSteelBlue, 0 , nullptr);
		m_commandList->GetComPtr()->ClearDepthStencilView(depthHandle, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);

		m_commandList->GetComPtr()->OMSetRenderTargets(1, &handle, true, &depthHandle);

		m_commandList->GetComPtr()->SetPipelineState(m_pipelineState->Get());

		m_commandList->GetComPtr()->IASetVertexBuffers(0, 1, &m_vertexData.VertexBufferView());
		m_commandList->GetComPtr()->IASetIndexBuffer(&m_vertexData.IndexBufferView());
		m_commandList->GetComPtr()->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

		m_commandList->GetComPtr()->DrawIndexedInstanced(3, 1, 0, 0, 0);

		m_commandList->TransitionResource(m_swapchain->GetCurrentBackBuffer()->Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);



		m_commandList->Close();
		m_commandQueue->ExecuteCommandList(*m_commandList);

		ThrowIfFailed(m_swapchain->GetComPointer()->Present(0, 0));
		m_swapchain->CurrentBufferIndex = (m_swapchain->CurrentBufferIndex + 1) % m_swapchain->BufferCount;

		currentFrameResource->EndFrame(*m_commandQueue);
	}
};


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance, PSTR cmdLine, int showCmd)
{
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
	try
	{
		AppTest app(hInstance);
		if (!app.Initialize())
			return 0;

		return app.Run();
	}
	catch (DxException& e)
	{
		MessageBox(nullptr, e.ToString().c_str(), L"HR Failed", MB_OK);
		return 0;
	}
}