#pragma once

#include "d3dx12.h"
#include <wrl/client.h>

namespace DX12Lib {

	class Device;

	template<typename T>
	class UploadBuffer
	{
	public:
		UploadBuffer(Device& device, UINT elementCount, bool isConstantBuffer);

		UploadBuffer(const UploadBuffer& rhs) = delete;
		UploadBuffer& operator=(const UploadBuffer& rhs) = delete;

		void Recreate() override;

		~UploadBuffer();

		void CopyData(int elementIndex, const T& data);


	private:
		Device& m_device;
		BYTE* m_mappedData = nullptr;
		UINT m_elementByteSize = 0;
		UINT m_elementCount = 0;
		bool m_isConstantBuffer = false;
		Microsoft::WRL::ComPtr<ID3D12Resource> m_resource;
	};

}




