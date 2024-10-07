#pragma once

#include "DX12Lib/DXWrapper/Resource.h"

namespace DX12Lib
{

    class UploadBuffer : public DX12Lib::Resource
    {
    public:
        virtual ~UploadBuffer() = default;

        void Create(size_t BufferSize);

        void* Map(void);
        void Unmap(size_t begin = 0, size_t end = -1);

        size_t GetBufferSize() const { return m_BufferSize; }

    protected:
        bool m_isMapped = false;
        size_t m_BufferSize;
    };
}




