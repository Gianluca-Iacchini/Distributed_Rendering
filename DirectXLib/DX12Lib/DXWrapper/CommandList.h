#pragma once

#include <d3d12.h>
#include <wrl/client.h>

namespace DX12Lib {

	class Device;
	class CommandAllocator;
	class PipelineState;
	class DescriptorHeap;

	class CommandList
	{
	public:
		CommandList(Device& device, CommandAllocator& commandAllocator, D3D12_COMMAND_LIST_TYPE = D3D12_COMMAND_LIST_TYPE_DIRECT, std::shared_ptr<PipelineState> pipelineState = nullptr);

		void SetDescriptorHeaps(std::vector<DescriptorHeap*> heaps);
		void SetDescriptorHeaps(DescriptorHeap** heaps, int size);
		void SetDescriptorHeap(DescriptorHeap* heap);

		void SetPipelineState(std::shared_ptr<PipelineState> pipelineState);

		void Reset(CommandAllocator& cmdAllocator);
		
		void Close();

		~CommandList();


	private:
		Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_commandList;
		std::shared_ptr<PipelineState> m_pipelineState;
		bool m_closed = false;

	public:
		const D3D12_COMMAND_LIST_TYPE m_cmdType;
	public:
		ID3D12GraphicsCommandList* Get() const { return m_commandList.Get(); }
		ID3D12GraphicsCommandList** GetAddressOf() { return m_commandList.GetAddressOf(); }
		Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> GetComPtr() const { return m_commandList; }

	};

}



