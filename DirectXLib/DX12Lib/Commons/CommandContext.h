#pragma once

#define MAX_RESOURCE_BARRIERS 16

#include <d3d12.h>
#include <mutex>
#include <queue>
#include <vector>

namespace DX12Lib {

	class Resource;
	class DescriptorHeap;
	class CommandAllocator;
	class CommandList;
	class DepthBuffer;
	class ColorBuffer;
	class Color;
	class PipelineState;
	class QueryHeap;
	class QueryHandle;
	class ReadBackBuffer;

	class CommandContext
	{
		friend class CommandContextManager;

	public:
		CommandContext(D3D12_COMMAND_LIST_TYPE type);
		virtual ~CommandContext();

		void InitializeApp();
		void Reset();

		D3D12_COMMAND_LIST_TYPE GetType() const { return m_type; }

		static CommandContext& Begin();
		void CopyBuffer(Resource& dest, Resource& src);
		void CopyBufferRegion(Resource& dest, size_t destOffset, Resource& src, size_t srcOffset, size_t numBytes);
		void SetPipelineState(PipelineState* pipelineState);
		void TransitionResource(Resource& res, D3D12_RESOURCE_STATES newState, bool transitionNow = false);
		void TransitionResource(ID3D12Resource* res, D3D12_RESOURCE_STATES beforeState, D3D12_RESOURCE_STATES newState, bool transitionNow = false);
		static void InitializeTexture(Resource& dest, UINT numSubresources, D3D12_SUBRESOURCE_DATA subresources[]);
		void SetDescriptorHeap(DescriptorHeap* heap);
		void SetDescriptorHeaps(std::vector<DescriptorHeap*> heaps);
		void BindDescriptorHeaps();

		void BeginQuery(QueryHeap& queryHeap, QueryHandle& handle, int offset = 0);
		void EndQuery(QueryHeap& queryHeap, QueryHandle& handle, int offset = 0);
		void ResolveQueryData(QueryHeap& queryHeap, QueryHandle& handle, ReadBackBuffer& destBuffer, UINT numQueries = 0);

		void FlushResourceBarriers();
		void InsertUAVBarrier(Resource* resource = nullptr, bool flushImmediate = false);
		void AddUAVIfNoBarriers(Resource* resource = nullptr, bool flushImmediate = false);
		UINT64 Flush(bool waitForCompletion = false);
		UINT64 Finish(bool waitForCompletion = false);


	public:
		CommandAllocator* m_currentAllocator = nullptr;
		CommandList* m_commandList = nullptr;


	private:
		CommandContext(const CommandContext&) = delete;
		CommandContext& operator=(const CommandContext&) = delete;

	protected:
		D3D12_COMMAND_LIST_TYPE m_type;
		D3D12_RESOURCE_BARRIER m_resourceBarrier[MAX_RESOURCE_BARRIERS];
		UINT m_numBarriersToFlush = 0;

		PipelineState* m_currentPipelineState = nullptr;
		DescriptorHeap* m_currentDescriptorHeaps[D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES];
	};

	class GraphicsContext : public CommandContext
	{
	public:
		void ClearColor(ColorBuffer& target, D3D12_RECT* rect);
		void ClearColor(ColorBuffer& target, float color[4], D3D12_RECT* rect);
		void ClearDepth(DepthBuffer& depthBuffer);
		void ClearDepthAndStencil(DepthBuffer& depthBuffer);
		void SetRenderTargets(UINT numRTVs, const D3D12_CPU_DESCRIPTOR_HANDLE rtvs[], D3D12_CPU_DESCRIPTOR_HANDLE dsv);
		void SetDepthStencilTarget(D3D12_CPU_DESCRIPTOR_HANDLE dsv);
		void SetViewportAndScissor(D3D12_VIEWPORT& viewport, D3D12_RECT& scissorRect);
		static void CommitGraphicsResources(D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT);
		static GraphicsContext& Begin();
	};

	class ComputeContext : public CommandContext
	{
	public:
		virtual ~ComputeContext() = default;
		static ComputeContext& Begin();
		void Dispatch(size_t groupCountX, size_t groupCountY = 1, size_t groupCountZ = 1);
		void Dispatch1D(size_t threadCount, size_t groupSize = 64);
		void Dispatch2D(size_t threadCountX, size_t threadCountY, size_t groupSizeX = 8, size_t groupSizeY = 8);
		void Dispatch3D(size_t threadCountX, size_t threadCountY, size_t threadCountZ, size_t groupSizeX = 8, size_t groupSizeY = 8, size_t groupSizeZ = 8);	
	};


	class CommandContextManager
	{
	public:
		CommandContextManager() {};
		~CommandContextManager() { DestroyAllContexts(); };
		CommandContext* AllocateContext(D3D12_COMMAND_LIST_TYPE type);
		void FreeContext(CommandContext* usedContext);
		void DestroyAllContexts();

	private:
		std::vector<std::unique_ptr<CommandContext>> m_contextPool[4];
		std::queue<CommandContext*> m_availableContexts[4];
		std::mutex m_contextAllocationMutex;
	};
}