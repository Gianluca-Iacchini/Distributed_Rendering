#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <wrl/client.h>
#include <Windows.h>
#include <d3d12.h>

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

    static std::wstring GetFileName(const std::wstring& filepath);
    static std::wstring GetFileDirectory(const std::wstring& filepath);
    static std::wstring GetWorkingDirectory();
    static void SetWorkingDirectory(const std::wstring& path);

    static std::wstring StartingWorkingDirectoryPath;
};











