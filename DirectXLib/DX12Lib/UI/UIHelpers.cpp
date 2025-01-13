#include "DX12Lib/pch.h"
#include "UIHelpers.h"

#include "DX12Lib/Commons/DX12Window.h"

#include "imgui.h"
#include "backends/imgui_impl_win32.h"
#include "backends/imgui_impl_dx12.h"

using namespace DX12Lib;
using namespace Graphics;

void imguiAlloc(D3D12_CPU_DESCRIPTOR_HANDLE* out_cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE* out_gpu_handle)
{
	auto descHandle = Renderer::s_textureHeap->Alloc(1);

	out_cpu_handle->ptr = descHandle.GetCPUPtr();
	out_gpu_handle->ptr = descHandle.GetGPUPtr();
}


void DX12Lib::UIHelpers::InitializeIMGUI(DX12Lib::DX12Window* window)
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	//ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

	ImGui::StyleColorsDark();

	ImGui_ImplWin32_Init(window->GetWindowHandle());

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
}

void DX12Lib::UIHelpers::ShutdownIMGUI()
{
	ImGui_ImplDX12_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();
}

void DX12Lib::UIHelpers::StartFrame()
{
	ImGui_ImplDX12_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();

}

bool DX12Lib::UIHelpers::OddIntegerSlider(const char* label, int* value, int min, int max)
{
	// Clamp the initial value to the nearest odd number
	if (*value % 2 == 0)
		*value += 1;

	int oddMin = min % 2 == 0? min + 1 : min;
	int oddMax = max % 2 == 0 ? max - 1 : max;

	int sliderValue = *value;
	*value = std::clamp(*value, oddMin, oddMax);


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

