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
#include "DX12Lib/GraphicsCore.h"
#include "DX12Lib/CommandContext.h"
#include "DX12Lib/RootSignature.h"

using namespace DirectX;
using namespace Microsoft::WRL;
using namespace Graphics;

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

	std::unordered_map<std::string, ComPtr<ID3D10Blob>> m_shaders;

	Keyboard keyboard;
	DirectX::Keyboard::KeyboardStateTracker tracker;

	RootSignature m_rootSignature;
	ComPtr<ID3D12PipelineState> m_pipelineState;
	//PipelineState m_pipelieState;

	//std::vector<std::unique_ptr<FrameResource>> m_frameResources;

	//FrameResource* m_currentFrameResource = nullptr;

	//UINT m_currentFrameResourceIndex = 0;
	//UINT gNumFrameResources = 3;

	//std::unordered_map<std::string, std::shared_ptr<Shader>> m_shaders;

	std::vector<D3D12_INPUT_ELEMENT_DESC> m_inputLayout;

	UINT64 frameFences[3] = { 0, 0, 0 };

public:
	AppTest(HINSTANCE hInstance) : D3DApp(hInstance) {};
	AppTest(const AppTest& rhs) = delete;
	AppTest& operator=(const AppTest& rhs) = delete;
	~AppTest() { 

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
		RootParameter constantRoot = RootParameter();
		constantRoot.InitAsConstants(0, 1);

		m_rootSignature = RootSignature(1, 0);
		m_rootSignature[0] = constantRoot;
		m_rootSignature.Finalize(D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
	}

	void BuildPSO()
	{
		//m_pipelieState = PipelineState();
		//m_pipelieState.InitializeDefaultStates();
		//m_pipelieState.SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
		//m_pipelieState.SetShader(m_shaders["basicVS"], ShaderType::Vertex);
		//m_pipelieState.SetShader(m_shaders["basicPS"], ShaderType::Pixel);

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

		ThrowIfFailed(s_device->GetComPtr()->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(m_pipelineState.GetAddressOf())));
	
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


		m_vertexData.VertexBufferGPU = Utils::CreateDefaultBuffer(s_device->GetComPtr(), context->m_commandList->GetComPtr(), triangleVertices, vbByteSize, m_vertexData.VertexBufferUploader);

		m_vertexData.IndexBufferGPU = Utils::CreateDefaultBuffer(s_device->GetComPtr(), context->m_commandList->GetComPtr(), triangleIndices, ibByteSize, m_vertexData.IndexBufferUploader);
	}

	virtual bool Initialize() override
	{
		if (!D3DApp::Initialize())
			return false;

		context->Reset();


		BuildEmptyRootSignature();
		BuildShadersAndInputLayout();
		BuildPSO();
		BuildVertexData();

		context->Finish();
		FlushCommandQueue();

		return true;
	}

	virtual void Update(const GameTime& gt) override
	{

		auto start = std::chrono::high_resolution_clock::now();
		auto kbState = keyboard.GetState();
		tracker.Update(kbState);

		if (tracker.IsKeyPressed(DirectX::Keyboard::Keys::Escape))
		{
			PostQuitMessage(0);
		}

		UINT64 currentFrame = frameFences[m_swapchain->CurrentBufferIndex];

		if (currentFrame != 0)
		{
			s_commandQueueManager->GetGraphicsQueue().WaitForFence(currentFrame);
		}
	}

	virtual void Draw(const GameTime& gt) override
	{
		context->Reset();

		context->m_commandList->GetComPtr()->RSSetViewports(1, &mScreenViewport);
		context->m_commandList->GetComPtr()->RSSetScissorRects(1, &mScissorRect);

		context->TransitionResource(CurrentBackBuffer(), D3D12_RESOURCE_STATE_RENDER_TARGET, true);

		float clearDepth = m_depthStencilBuffer->GetClearDepth();
		float clearStencil = m_depthStencilBuffer->GetClearStencil();

		context->m_commandList->GetComPtr()->ClearRenderTargetView(CurrentBackBufferView(), DirectX::Colors::LightSteelBlue, 0, nullptr);
		context->m_commandList->GetComPtr()->ClearDepthStencilView(DepthStencilView(), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, clearDepth, clearStencil, 0, nullptr);

		context->m_commandList->GetComPtr()->OMSetRenderTargets(1, &CurrentBackBufferView(), true, &DepthStencilView());

		context->m_commandList->GetComPtr()->SetGraphicsRootSignature(m_rootSignature.Get());

		float time = gt.TotalTime();

		context->m_commandList->GetComPtr()->SetGraphicsRoot32BitConstants(0, 1, &time, 0);

		context->m_commandList->GetComPtr()->SetPipelineState(m_pipelineState.Get());

		context->m_commandList->GetComPtr()->IASetVertexBuffers(0, 1, &m_vertexData.VertexBufferView());
		context->m_commandList->GetComPtr()->IASetIndexBuffer(&m_vertexData.IndexBufferView());
		context->m_commandList->GetComPtr()->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

		context->m_commandList->GetComPtr()->DrawIndexedInstanced(3, 1, 0, 0, 0);

		context->TransitionResource(CurrentBackBuffer(), D3D12_RESOURCE_STATE_PRESENT, true);

		frameFences[m_swapchain->CurrentBufferIndex] = context->Finish();



		ThrowIfFailed(m_swapchain->GetComPointer()->Present(0, DXGI_PRESENT_ALLOW_TEARING));

		m_swapchain->CurrentBufferIndex = (m_swapchain->CurrentBufferIndex + 1) % m_swapchain->BufferCount;
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