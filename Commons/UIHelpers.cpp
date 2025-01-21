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
#endif // CVGI_DX12

void Commons::UIHelpers::InitializeIMGUI(HWND windowHandle)
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	//ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

	ImGui::StyleColorsDark();

#if CVGI_DX12
	ImGui_ImplWin32_Init(windowHandle);

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

#endif
}

void Commons::UIHelpers::ShutdownIMGUI()
{
#if CVGI_DX12
	ImGui_ImplDX12_Shutdown();
#endif
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();
}

void Commons::UIHelpers::StartFrame()
{
#if CVGI_DX12
	ImGui_ImplDX12_NewFrame();
#endif
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();

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

