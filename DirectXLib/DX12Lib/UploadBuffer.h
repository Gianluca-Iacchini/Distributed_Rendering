
#include "Helpers.h"
#include "Resource.h"


#ifndef UPLOADBUFFER_H
#define UPLOADBUFFER_H

template<typename T>
class UploadBuffer : public Resource
{
public:
	UploadBuffer(Device& device, UINT elementCount, bool isConstantBuffer);

	UploadBuffer(const UploadBuffer& rhs) = delete;
	UploadBuffer& operator=(const UploadBuffer& rhs) = delete;

	void Recreate() override;

	~UploadBuffer();

	void CopyData(int elementIndex, const T& data);


private:
	BYTE* m_mappedData = nullptr;
	UINT m_elementByteSize = 0;
	UINT m_elementCount = 0;
	bool m_isConstantBuffer = false;

};

#endif // !UPLOADBUFFER_H




