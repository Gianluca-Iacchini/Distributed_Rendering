#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <wrl/client.h>
#include <Windows.h>
#include <d3d12.h>
#include <assert.h>
#include <queue>
#include "DX12Lib/DXWrapper/Resource.h"
#include "mutex"


// From Microsoft mini engine
#define D3D12_GPU_VIRTUAL_ADDRESS_NULL    ((D3D12_GPU_VIRTUAL_ADDRESS)0)
#define D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN ((D3D12_GPU_VIRTUAL_ADDRESS)-1)
#define CONSTANT_BUFFER_SIZE 256

extern const int gNumFrameResources;


inline std::string HrToString(HRESULT hr)
{
    char s_str[64] = {};
    sprintf_s(s_str, "HRESULT of 0x%08X", static_cast<UINT>(hr));
    return std::string(s_str);
}

class DxException 
{
public:
    DxException() = default;
    DxException(HRESULT hr, const std::wstring& functionName, const std::wstring& filename, int lineNumber);
    std::wstring ToString() const;

    HRESULT ErrorCode = S_OK;
    std::wstring FunctionName;
    std::wstring Filename;
    int LineNumber = -1;
};

inline std::wstring AnsiToWstring(const std::string& str)
{
    WCHAR  buffer[512];
    MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, buffer, 512);
    return std::wstring(buffer);
}

inline void Print(const char* msg) { printf("%s", msg); }
inline void Print(const wchar_t* msg) { wprintf(L"%ws", msg); }

inline void PrintSubMessage(const char* format, ...)
{
    Print("--> ");
    char buffer[256];
    va_list ap;
    va_start(ap, format);
    vsprintf_s(buffer, 256, format, ap);
    va_end(ap);
    Print(buffer);
    Print("\n");
}
inline void PrintSubMessage(const wchar_t* format, ...)
{
    Print("--> ");
    wchar_t buffer[256];
    va_list ap;
    va_start(ap, format);
    vswprintf(buffer, 256, format, ap);
    va_end(ap);
    Print(buffer);
    Print("\n");
}

inline void PrintSubMessage(void)
{
}



#define STRINGIFY(x) #x
#define STRINGIFY_BUILTIN(x) STRINGIFY(x)

#ifndef ThrowIfFailed
#define ThrowIfFailed( hr, ... ) \
        if (FAILED(hr)) { \
            Print("\nHRESULT failed in " STRINGIFY_BUILTIN(__FILE__) " @ " STRINGIFY_BUILTIN(__LINE__) "\n"); \
            PrintSubMessage("hr = 0x%08X", hr); \
            PrintSubMessage(__VA_ARGS__); \
            Print("\n"); \
            fflush(stdout); \
            __debugbreak(); \
        }
#endif



class Utils
{
public:
    /// <summary>
    /// Rounds up to the nearest multiple of alignAt
    /// </summary>
    /// <param name="byteSize">Input bytesize</param>
    /// <param name="alignAt">Value to round up to. Must be a power of 2.</param>
    /// <returns></returns>
    static UINT AlignAtBytes(UINT byteSize, UINT alignAt)
    {
		return (byteSize + alignAt - 1) & ~(alignAt-1);
	}

    static Microsoft::WRL::ComPtr<ID3DBlob> Compile(const std::wstring& filename,
        const D3D_SHADER_MACRO* defines,
        const std::string& entryPoint,
        const std::string& target);

    static Microsoft::WRL::ComPtr<ID3D12Resource> CreateDefaultBuffer(const void* initData, UINT64 byteSize);

    static inline std::string ToString(const std::wstring& wstr)
    {
        std::string str;
        size_t size;
        str.resize(wstr.length());
        wcstombs_s(&size, &str[0], str.size() + 1, wstr.c_str(), wstr.size());
        return str;
    }

    static inline std::wstring ToWstring(const char* stringLiteral)
    {

        std::string sourceDirNarrow(stringLiteral);

        // Convert to std::wstring using wstringstream
        std::wstringstream wss;
        wss << sourceDirNarrow.c_str();
        std::wstring sourceDirWide = wss.str();

        return sourceDirWide;
    }

    static inline std::wstring ToWstring(const std::string& str)
    {
        WCHAR buffer[512];
        MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, buffer, 512);
        return std::wstring(buffer);
    }

    static inline std::wstring GUIDToWstring(const GUID& guid)
	{
		WCHAR buffer[64];
		UINT size = StringFromGUID2(guid, buffer, 64);
		
        assert(size > 0 && size < 64 && "GUIDToWstring failed");
        
        return std::wstring(buffer);
	}

    static std::wstring GetFileName(const std::wstring& filepath);
    static std::wstring GetFileDirectory(const std::wstring& filepath);
    static std::wstring GetWorkingDirectory();
    static void SetWorkingDirectory(const std::wstring& path);

    static std::wstring StartingWorkingDirectoryPath;
};

namespace DX12Lib {
    template <typename T>
    class ThreadSafeQueue
    {
    public:
        ThreadSafeQueue() : m_maxSize(0) {}
        ThreadSafeQueue(std::uint16_t maxSize) : m_maxSize(maxSize) {}
        ~ThreadSafeQueue() = default;

        void SetMaxSize(std::uint16_t maxSize) { m_maxSize = maxSize; }
        void GetMaxSize() { return m_maxSize; }

        bool EvictPush(T item, T& evictedItem);
        void Push(T item);
        bool Pop(T& outPop);

        std::queue<T>& GetQueue() { return m_queue; }

    private:
        std::uint16_t m_maxSize = 0;
        std::queue<T> m_queue;
        std::mutex m_mutex;
    };


    template<typename T>
    inline void ThreadSafeQueue<T>::Push(T item)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_queue.size() < m_maxSize)
        {
            m_queue.push(item);
        }

    }

    template<typename T>
    inline bool ThreadSafeQueue<T>::EvictPush(T item, T& evictedItem)
    {
        bool evicted = false;

        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_maxSize > 0 && m_queue.size() >= m_maxSize)
        {
            evictedItem = m_queue.front();
            m_queue.pop();
            evicted = true;
        }

        m_queue.push(item);

        return evicted;
    }

    template<typename T>
    inline bool ThreadSafeQueue<T>::Pop(T& outPop)
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_queue.empty())
        {
            return false;
        }

        T value = m_queue.front();

        outPop = value;
        m_queue.pop();
        return true;

    }

    template<typename T>
    class ReusableQueue {
    public:
        ReusableQueue() {}

        void Push(T element) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_elementQueue.push(element);
            m_cv.notify_one();
        }

        void GetFromPool(T& element) {
            std::unique_lock<std::mutex> lock(m_mutex);

            // If the flag is set, we wait for an element to be available in the pool
            if (ShouldWait)
            {
                m_cv.wait(lock, [this] { return !m_availableElementsPool.empty() || m_done; });
            }
            // If the flag is not set and the pool is empty, we return the oldest element in the queue (the one at the front) to the pool 
            else if (m_availableElementsPool.empty())
            {
                m_availableElementsPool.push_back(m_elementQueue.front());
                m_elementQueue.pop();
            }

			// Usually needed only when the thread has to be abruptly stopped
            if (m_done)
                return;

            element = m_availableElementsPool.back();
            m_availableElementsPool.pop_back();
        }

        void Pop(T& element) {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this] { return !m_elementQueue.empty() || m_done; });

            // Usually needed only when the thread has to be abruptly stopped
            if (m_done)
                return;

            element = m_elementQueue.front();
            m_elementQueue.pop();
        }

        bool NonBlockingPop(T& element) {
			std::unique_lock<std::mutex> lock(m_mutex);

			if (m_elementQueue.empty())
				return false;

			element = m_elementQueue.front();
			m_elementQueue.pop();
			return true;
        }

		void AddNewElementToPool(T element) {
			std::lock_guard<std::mutex> lock(m_mutex);
			m_availableElementsPool.push_back(element);
		}

        void ReturnToPool(T& element) {
            std::lock_guard<std::mutex> lock(m_mutex);
            
			m_availableElementsPool.push_back(element);
        }

        void SetDone() {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_done = true;
            m_cv.notify_all();
        }

        unsigned int GetQueueSize() {
            std::unique_lock<std::mutex> lock(m_mutex);
            return m_elementQueue.size();
        }

        unsigned int GetPoolSize() {
            std::unique_lock<std::mutex> lock(m_mutex);
            return m_availableElementsPool.size();
        }

		std::mutex& GetMutex() {
			return m_mutex;
		}

		std::queue<T>& GetQueue() {
			return m_elementQueue;
		}

		std::vector<T>& GetPool() { return m_availableElementsPool; }

    public:
        bool ShouldWait = true;

    private:
        int m_maxElements = 0;
        std::queue<T> m_elementQueue;
        std::vector<T> m_availableElementsPool;
        std::mutex m_mutex;
        std::condition_variable m_cv;
        bool m_done = false;
    };
}