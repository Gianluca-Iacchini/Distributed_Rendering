#include "UIHelpers.h"

#include "imgui.h"
#include "backends/imgui_impl_win32.h"

#if CVGI_DX12
#include "backends/imgui_impl_dx12.h"

using namespace DX12Lib;
using namespace Graphics;

#include "backends/imgui_impl_dx12.h"

void imguiAlloc(D3D12_CPU_DESCRIPTOR_HANDLE* out_cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE* out_gpu_handle)
{
	auto descHandle = Renderer::s_textureHeap->Alloc(1);

	out_cpu_handle->ptr = descHandle.GetCPUPtr();
	out_gpu_handle->ptr = descHandle.GetGPUPtr();
}
#elif CVGI_GL
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#endif 



void Commons::UIHelpers::InitializeIMGUI(IMGUIWND imguiWnd)
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	//ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

	ImGui::StyleColorsDark();

#if CVGI_DX12
	ImGui_ImplWin32_Init(imguiWnd);

	ImGui_ImplDX12_InitInfo initInfo = {};
	initInfo.Device = Graphics::s_device->Get();
	initInfo.CommandQueue = Graphics::s_commandQueueManager->GetGraphicsQueue().Get();
	initInfo.NumFramesInFlight = 3;
	initInfo.RTVFormat = Graphics::m_backBufferFormat;
	initInfo.DSVFormat = Graphics::m_depthStencilFormat;
	initInfo.SrvDescriptorHeap = Renderer::s_textureHeap->Get();
	initInfo.SrvDescriptorAllocFn = [](ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE* out_cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE* out_gpu_handle) { return imguiAlloc(out_cpu_handle, out_gpu_handle); };
	initInfo.SrvDescriptorFreeFn = [](ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE out_cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE out_gpu_handle) {};


	ImGui_ImplDX12_Init(&initInfo);

#elif CVGI_GL
	ImGui_ImplGlfw_InitForOpenGL(imguiWnd, true);
	ImGui_ImplOpenGL3_Init("#version 330");
#endif
}

void Commons::UIHelpers::ShutdownIMGUI()
{
#if CVGI_DX12
	ImGui_ImplDX12_Shutdown();
	ImGui_ImplWin32_Shutdown();
#elif CVGI_GL
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
#endif

	ImGui::DestroyContext();
}

void Commons::UIHelpers::StartFrame()
{
#if CVGI_DX12
	ImGui_ImplDX12_NewFrame();
	ImGui_ImplWin32_NewFrame();
#elif CVGI_GL
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
#endif

	ImGui::NewFrame();

}

void Commons::UIHelpers::EndFrame()
{
	ImGui::EndFrame();
}

void Commons::UIHelpers::ControlInfoBlock(bool isConnected)
{
	ImGui::SeparatorText("Controls");

	float maxX = ImGui::CalcTextSize("- Hold Right Mouse Button:\t").x;

	if (!isConnected)
	{
		ImGui::Text("- W, A, S, D:");
		ImGui::SameLine(maxX);
		ImGui::Text("Move Camera");

		ImGui::Text("- E, Q:");
		ImGui::SameLine(maxX);
		ImGui::Text("Move Camera Up/Down");

		ImGui::Text("- Hold Right Mouse Button:");
		ImGui::SameLine(maxX);
		ImGui::Text("Rotate Camera");

		ImGui::Text("- Arrow Keys:");
		ImGui::SameLine(maxX);
		ImGui::Text("Move Light");
	}
	else
	{
		ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Client is connected. Scene control is disabled.");
	}

	ImGui::Text("- ESC:");
	ImGui::SameLine(maxX);
	ImGui::Text("Quit");
}

void Commons::UIHelpers::ConnectedClient(const char* peerAddr, UINT32 ping)
{

	ImGui::BeginTable("ClientTable", 3, ImGuiTableFlags_BordersInner | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_RowBg);
	ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed);
	ImGui::TableSetupColumn("Client Address", ImGuiTableColumnFlags_WidthStretch);
	ImGui::TableSetupColumn("Ping", ImGuiTableColumnFlags_WidthFixed);
	ImGui::TableHeadersRow();

	ImGui::TableNextRow();
	ImGui::TableNextColumn();
	ImGui::Text("Client %d\t", 0);
	ImGui::TableNextColumn();
	ImGui::Text("%s", peerAddr);
	ImGui::TableNextColumn();
	ImGui::Text("%d ms", ping);

	ImGui::EndTable();
	
}

bool Commons::UIHelpers::OddIntegerSlider(const char* label, int* value, int min, int max)
{
	// Clamp the initial value to the nearest odd number
	if (*value % 2 == 0)
		*value += 1;

	int oddMin = min % 2 == 0? min + 1 : min;
	int oddMax = max % 2 == 0 ? max - 1 : max;

	int sliderValue = *value;

	if (*value < oddMin)
		*value = oddMin;
	else if (*value > oddMax)
		*value = oddMax;


	if (ImGui::SliderInt(label, value, oddMin, oddMax))
	{
		if (*value % 2 == 0)
		{
			*value += 1;
		}

		return true;
	}

	return false;
}

