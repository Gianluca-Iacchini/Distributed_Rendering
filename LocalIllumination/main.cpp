#include "DX12Lib/Helpers.h"
#include "DX12Lib/D3DApp.h"
#include <iostream>
#include "DX12Lib/CIDXGIFactory.h"
#include "Keyboard.h"
#include "DX12Lib/Shader.h"
#include "Dx12Lib/PipelineState.h"
#include "DX12Lib/Helpers.h"
#include "FrameResource.h"
#include "DX12Lib/Fence.h"
#include "DX12Lib/CommandQueue.h"
#include "DX12Lib/CommandList.h"
#include "DX12Lib/CommandAllocator.h"
#include "DX12Lib/Swapchain.h"
#include "DX12Lib/DepthBuffer.h"

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

	Keyboard keyboard;
	DirectX::Keyboard::KeyboardStateTracker tracker;

	ComPtr<ID3D12RootSignature> m_rootSignature;
	ComPtr<ID3D12PipelineState> m_pipelineState;

	std::vector<std::unique_ptr<FrameResource>> m_frameResources;

	FrameResource* m_currentFrameResource = nullptr;

	UINT m_currentFrameResourceIndex = 0;
	UINT gNumFrameResources = 3;

	std::unordered_map<std::string, ComPtr<ID3D10Blob>> m_shaders;

	std::vector<D3D12_INPUT_ELEMENT_DESC> m_inputLayout;

	ComPtr<ID3D12CommandQueue> mCommandQueue;
	ComPtr<ID3D12GraphicsCommandList> mCommandList;
	ComPtr<ID3D12CommandAllocator> mCommandListAllocator;

public:
	AppTest(HINSTANCE hInstance) : D3DApp(hInstance) {};
	AppTest(const AppTest& rhs) = delete;
	AppTest& operator=(const AppTest& rhs) = delete;
	~AppTest() { 
		if (m_device != nullptr)
			FlushCommandQueue();
		
		FreeConsole();
	};

	void BuildShadersAndInputLayout()
	{
		std::wstring srcDir = ToWstring(SOURCE_DIR);
		std::wstring shaderFile = srcDir + L"\\Shaders\\Basic.hlsl";

		m_shaders["basicVS"] = Utils::Compile(shaderFile, nullptr, "VS", "vs_5_1");
		m_shaders["basicPS"] = Utils::Compile(shaderFile, nullptr, "PS", "ps_5_1");

		m_inputLayout =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
		};

	}

	void BuildEmptyRootSignature()
	{
		CD3DX12_ROOT_PARAMETER slotRootParameter[1];

		// Create a single descriptor table of CBVs.
		CD3DX12_DESCRIPTOR_RANGE cbvTable;
		cbvTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0);
		slotRootParameter[0].InitAsConstants(1, 0, 0, D3D12_SHADER_VISIBILITY_ALL);

		// A root signature is an array of root parameters.
		CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(1, slotRootParameter, 0, nullptr,
			D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

		// create a root signature with a single slot which points to a descriptor range consisting of a single constant buffer
		ComPtr<ID3DBlob> serializedRootSig = nullptr;
		ComPtr<ID3DBlob> errorBlob = nullptr;
		HRESULT hr = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
			serializedRootSig.GetAddressOf(), errorBlob.GetAddressOf());

		if (errorBlob != nullptr)
		{
			::OutputDebugStringA((char*)errorBlob->GetBufferPointer());
		}
		ThrowIfFailed(hr);

		ThrowIfFailed(m_device->GetComPtr()->CreateRootSignature(
			0,
			serializedRootSig->GetBufferPointer(),
			serializedRootSig->GetBufferSize(),
			IID_PPV_ARGS(&m_rootSignature)));
	}

	void BuildPSO()
	{
		D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
		ZeroMemory(&psoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));

		psoDesc.InputLayout = { m_inputLayout.data(), (UINT)m_inputLayout.size()};
		psoDesc.pRootSignature = m_rootSignature.Get();
		psoDesc.VS = { reinterpret_cast<BYTE*>(m_shaders["basicVS"]->GetBufferPointer()), m_shaders["basicVS"]->GetBufferSize() };
		psoDesc.PS = { reinterpret_cast<BYTE*>(m_shaders["basicPS"]->GetBufferPointer()), m_shaders["basicPS"]->GetBufferSize() };
		psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
		psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
		psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
		psoDesc.SampleMask = UINT_MAX;
		psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
		psoDesc.NumRenderTargets = 1;
		psoDesc.RTVFormats[0] = mBackBufferFormat;
		psoDesc.SampleDesc.Count = m_4xMsaaState ? 4 : 1;
		psoDesc.SampleDesc.Quality = m_4xMsaaState ? (m_4xMsaaQuality - 1) : 0;
		psoDesc.DSVFormat = mDepthStencilFormat;

		ThrowIfFailed(m_device->GetComPtr()->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(m_pipelineState.GetAddressOf())));
	
	}

	void BuildVertexData()
	{
		std::uint16_t triangleIndices[3] = { 0, 1, 2 };

		const UINT vbByteSize = (UINT)sizeof(triangleVertices);
		const UINT ibByteSize = 3 * (UINT)sizeof(std::uint16_t);

		ZeroMemory(&m_vertexData, sizeof(VertexResourceData));

		ThrowIfFailed(D3DCreateBlob(vbByteSize, &m_vertexData.VertexBufferCPU));
		CopyMemory(m_vertexData.VertexBufferCPU->GetBufferPointer(), triangleVertices, vbByteSize);

		ThrowIfFailed(D3DCreateBlob(ibByteSize, &m_vertexData.IndexBufferCPU));
		CopyMemory(m_vertexData.IndexBufferCPU->GetBufferPointer(), triangleIndices, ibByteSize);

		m_vertexData.VertexBufferByteSize = vbByteSize;
		m_vertexData.IndexBufferByteSize = ibByteSize;

		m_vertexData.VertexBufferStride = sizeof(Vertex);
		m_vertexData.IndexBufferFormat = DXGI_FORMAT_R16_UINT;

		m_vertexData.VertexBufferGPU = Utils::CreateDefaultBuffer(m_device->GetComPtr(), mCommandList, triangleVertices, vbByteSize, m_vertexData.VertexBufferUploader);

		m_vertexData.IndexBufferGPU = Utils::CreateDefaultBuffer(m_device->GetComPtr(), mCommandList, triangleIndices, ibByteSize, m_vertexData.IndexBufferUploader);
	}

	virtual bool Initialize() override
	{
		if (!D3DApp::Initialize())
			return false;

		mCommandQueue = m_commandQueue->GetComPtr();
		mCommandList = m_commandList->GetComPtr();
		mCommandListAllocator = m_appCommandAllocator->GetComPtr();

		m_commandList->Reset(*m_appCommandAllocator);

		for (int i = 0; i < gNumFrameResources; ++i)
		{
			m_frameResources.push_back(std::make_unique<FrameResource>(*m_device));
		}

		BuildEmptyRootSignature();
		BuildShadersAndInputLayout();
		BuildPSO();
		BuildVertexData();

		m_commandList->Close();

		m_commandQueue->ExecuteCommandList(*m_commandList);

		FlushCommandQueue();

		

		return true;
	}

	virtual void Update(const GameTime& gt) override
	{


		auto kbState = keyboard.GetState();
		tracker.Update(kbState);

		if (tracker.IsKeyPressed(DirectX::Keyboard::Keys::Escape))
		{
			PostQuitMessage(0);
		}


		m_currentFrameResourceIndex = (m_currentFrameResourceIndex + 1) % gNumFrameResources;
		m_currentFrameResource = m_frameResources[m_currentFrameResourceIndex].get();

		if (m_currentFrameResource->Fence != 0)
		{
			m_appFence->WaitForFence(m_currentFrameResource->Fence);
		}
	}

	virtual void Draw(const GameTime& gt) override
	{

		auto cmdListAlloc = m_frameResources[m_currentFrameResourceIndex]->CmdListAlloc;

		cmdListAlloc->Reset();


		m_commandList->Reset(*cmdListAlloc);


		m_commandList->GetComPtr()->RSSetViewports(1, &mScreenViewport);
		m_commandList->GetComPtr()->RSSetScissorRects(1, &mScissorRect);

		m_commandList->TransitionResource(CurrentBackBuffer().Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);

		float clearDepth = m_depthStencilBuffer->GetClearDepth();
		float clearStencil = m_depthStencilBuffer->GetClearStencil();

		m_commandList->GetComPtr()->ClearRenderTargetView(CurrentBackBufferView(), DirectX::Colors::LightSteelBlue, 0, nullptr);
		m_commandList->GetComPtr()->ClearDepthStencilView(DepthStencilView(), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, clearDepth, clearStencil, 0, nullptr);

		m_commandList->GetComPtr()->OMSetRenderTargets(1, &CurrentBackBufferView(), true, &DepthStencilView());

		m_commandList->GetComPtr()->SetGraphicsRootSignature(m_rootSignature.Get());

		float time = gt.TotalTime();

		m_commandList->GetComPtr()->SetGraphicsRoot32BitConstants(0, 1, &time, 0);

		m_commandList->GetComPtr()->SetPipelineState(m_pipelineState.Get());

		m_commandList->GetComPtr()->IASetVertexBuffers(0, 1, &m_vertexData.VertexBufferView());
		m_commandList->GetComPtr()->IASetIndexBuffer(&m_vertexData.IndexBufferView());
		m_commandList->GetComPtr()->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

		m_commandList->GetComPtr()->DrawIndexedInstanced(3, 1, 0, 0, 0);

		m_commandList->TransitionResource(CurrentBackBuffer().Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);


		m_commandList->Close();

		m_commandQueue->ExecuteCommandList(*m_commandList);
		ThrowIfFailed(m_swapchain->GetComPointer()->Present(0,0));
		m_swapchain->CurrentBufferIndex = (m_swapchain->CurrentBufferIndex + 1) % m_swapchain->BufferCount;

		m_currentFrameResource->Fence = ++m_appFence->FenceValue;

		m_commandQueue->Signal(m_appFence.get());
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