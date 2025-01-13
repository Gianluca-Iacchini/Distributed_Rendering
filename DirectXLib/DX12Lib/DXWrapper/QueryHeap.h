#pragma once

namespace DX12Lib
{
	class QueryHandle
	{
	public:
		QueryHandle(UINT index, UINT numQueries) : m_index(index), m_numQueries(numQueries)
		{
		}

		QueryHandle() { m_index = 0; m_numQueries = 0; }

		UINT GetIndex() const { return m_index; }
		UINT GetSize() const { return m_numQueries; }

		operator UINT() const { return m_index; }

		void QueryStarted() { ++m_queryInFlight; }
		void QueryEnded() { --m_queryInFlight; }

		bool IsValid() const { return m_queryInFlight == 0; }

		QueryHandle& operator =(const QueryHandle& other)
		{
			m_index = other.m_index;
			m_numQueries = other.m_numQueries;
			m_queryInFlight = other.m_queryInFlight;

			return *this;
		}

	private:
		UINT m_queryInFlight = 0;
		UINT m_index = 0;
		UINT m_numQueries = 0;
	};

	class QueryHeap
	{
	public:
		QueryHeap() {}
		void Create(D3D12_QUERY_HEAP_TYPE type, UINT16 maxElements);
		QueryHandle Alloc(UINT count);
		void Free(QueryHandle handle);
		D3D12_QUERY_HEAP_TYPE GetType() const { return m_type; }

	private:
		int GetAllocIndex(UINT allocSize);
		void FreeRegion(UINT index, UINT count);

	public:
		Microsoft::WRL::ComPtr<ID3D12QueryHeap> GetComPtr() { return m_queryHeap; }
		ID3D12QueryHeap* Get() { return m_queryHeap.Get(); }
		ID3D12QueryHeap* const* GetAddressOf() { return m_queryHeap.GetAddressOf(); }
		operator ID3D12QueryHeap* () { return m_queryHeap.Get(); }

	private:
		Microsoft::WRL::ComPtr<ID3D12QueryHeap> m_queryHeap;
		D3D12_QUERY_HEAP_TYPE m_type;
		UINT m_maxElements = 0;

		std::vector<UINT64> m_occupancyBitmap;
	};
}

