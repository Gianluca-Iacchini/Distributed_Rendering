#pragma once
#define WIN32_LEAN_AND_MEAN
#include <Windows.h> // For HRESULT
#include <stdexcept>

#include <sstream>
#include <dxgi1_6.h>
#include <directx/d3dx12.h>
#include <cassert>
#include <string>
#include <unordered_map>
#include <DirectXColors.h>
#include <DirectXCollision.h>
#include <d3dcompiler.h>
#include <array>
#include "MathHelper.h"

extern const int gNumFrameResources;

inline std::wstring ToWstring(const char* stringLiteral) 
{

    std::string sourceDirNarrow(stringLiteral);

    // Convert to std::wstring using wstringstream
    std::wstringstream wss;
    wss << sourceDirNarrow.c_str();
    std::wstring sourceDirWide = wss.str();

    return sourceDirWide;
}

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

inline std::wstring AnsiToWString(const std::string& str)
{
    WCHAR buffer[512];
	MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, buffer, 512);
	return std::wstring(buffer);
}

#ifndef ThrowIfFailed
#define ThrowIfFailed(x)                                              \
{                                                                     \
    HRESULT hr__ = (x);                                               \
    std::wstring wfn = AnsiToWString(__FILE__);                       \
    if(FAILED(hr__)) { throw DxException(hr__, L#x, wfn, __LINE__); } \
}
#endif

#define CONSTANT_BUFFER_SIZE 256

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

    static Microsoft::WRL::ComPtr<ID3D12Resource> CreateDefaultBuffer(Microsoft::WRL::ComPtr<ID3D12Device>& device,
        		Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>& cmdList,
        		const void* initData,
        		UINT64 byteSize,
        		Microsoft::WRL::ComPtr<ID3D12Resource>& uploadBuffer);
};

struct SubmeshGeometry
{
    UINT indexCount = 0;
    UINT startIndexLocation = 0;
    INT baseVertexLocation = 0;

    DirectX::BoundingBox bounds;
};

struct MeshGeometry
{
    std::string Name;

    Microsoft::WRL::ComPtr<ID3DBlob> VertexBufferCPU = nullptr;
    Microsoft::WRL::ComPtr<ID3DBlob> IndexBufferCPU = nullptr;

    Microsoft::WRL::ComPtr<ID3D12Resource> VertexBufferGPU = nullptr;
    Microsoft::WRL::ComPtr<ID3D12Resource> IndexBufferGPU = nullptr;

    Microsoft::WRL::ComPtr<ID3D12Resource> VertexBufferUploader = nullptr;
    Microsoft::WRL::ComPtr<ID3D12Resource> IndexBufferUploader = nullptr;

    UINT VertexByteStride = 0;
    UINT VertexBufferByteSize = 0;

    DXGI_FORMAT IndexFormat = DXGI_FORMAT_R16_UINT;
    UINT IndexBufferByteSize = 0;

    std::unordered_map<std::string, SubmeshGeometry> DrawArgs;

    D3D12_VERTEX_BUFFER_VIEW VertexBufferView() const
    {
        D3D12_VERTEX_BUFFER_VIEW vbv;
        vbv.BufferLocation = VertexBufferGPU->GetGPUVirtualAddress();
        vbv.SizeInBytes = VertexBufferByteSize;
        vbv.StrideInBytes = VertexByteStride;

        return vbv;
	}

    D3D12_INDEX_BUFFER_VIEW IndexBufferView() const
    {
		D3D12_INDEX_BUFFER_VIEW ibv;
		ibv.BufferLocation = IndexBufferGPU->GetGPUVirtualAddress();
		ibv.Format = IndexFormat;
		ibv.SizeInBytes = IndexBufferByteSize;

        return ibv;
	}

    void DisposeUploaders()
    {
		VertexBufferUploader = nullptr;
		IndexBufferUploader = nullptr;
	}
};

struct MaterialConstants
{
    DirectX::XMFLOAT4 DiffuseAlbedo = { 1.0f, 1.0f, 1.0f, 1.0f };
    DirectX::XMFLOAT3 FresnelR0 = { 0.01f, 0.01f, 0.01f };
    float Roughness = .25f;

    DirectX::XMFLOAT4X4 MatTransform = MathHelper::Identity4x4();;
};

struct Material
{
    std::string Name;

    int MatCBIndex = -1;

    int DiffuseSrvHeapIndex = -1;

    int NormalSrvHeapIndex = -1;

    int NumFramesDirty = gNumFrameResources;

    DirectX::XMFLOAT4 DiffuseAlbedo = { 1.0f, 1.0f, 1.0f, 1.0f };
    DirectX::XMFLOAT3 FresnelR0 = { 0.01f, 0.01f, 0.01f };
    float Roughness = 0.25f;
    DirectX::XMFLOAT4X4 MatTransform = MathHelper::Identity4x4();
};

struct Texture
{
    std::string Name;

    std::wstring Filename;

    Microsoft::WRL::ComPtr<ID3D12Resource> Resource = nullptr;
    Microsoft::WRL::ComPtr<ID3D12Resource> UploadHeap = nullptr;
};

#define MaxLights 16

struct Light
{
    DirectX::XMFLOAT3 Strength = { 0.5f, 0.5f, 0.5f };
    float fallOffStart = 1.0f; // point/spot light only
    DirectX::XMFLOAT3 Direction = { 0.0f, -1.0f, 0.0f };
    float fallOffEnd = 10.0f; // point/spot light only
    DirectX::XMFLOAT3 Position = { 0.0f, 0.0f, 0.0f };
    float SpotPower = 64.0f; // spot light only
};


