#include "DX12Lib/pch.h"
#include <comdef.h>
#include "GraphicsMemory.h"
#include "ResourceUploadBatch.h"


using Microsoft::WRL::ComPtr;

DxException::DxException(HRESULT hr, const std::wstring& functionName, const std::wstring& filename, int lineNumber)
	: ErrorCode(hr),
	FunctionName(functionName),
	Filename(filename),
	LineNumber(lineNumber)
{
}

std::wstring DxException::ToString() const
{
	// Get the string description of the error code.
	_com_error err(ErrorCode);
	std::wstring msg = err.ErrorMessage();

	return FunctionName + L" failed in " + Filename + L"; line " + std::to_wstring(LineNumber) + L"; error: " + msg;
}

Microsoft::WRL::ComPtr<ID3DBlob> Utils::Compile(const std::wstring& filename, const D3D_SHADER_MACRO* defines, const std::string& entryPoint, const std::string& target)
{
	UINT compileFlags = 0;

#if defined(DEBUG) || defined(_DEBUG)
	compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

	HRESULT hr = S_OK;

	ComPtr<ID3DBlob> byteCode = nullptr;
	ComPtr<ID3DBlob> errors = nullptr;

	hr = D3DCompileFromFile(filename.c_str(), defines, D3D_COMPILE_STANDARD_FILE_INCLUDE, entryPoint.c_str(), target.c_str(), compileFlags, 0, byteCode.GetAddressOf(), errors.GetAddressOf());

	if (errors != nullptr)
	{
		OutputDebugStringA((char*)errors->GetBufferPointer());
	}

	ThrowIfFailed(hr);

	return byteCode;
}

Microsoft::WRL::ComPtr<ID3D12Resource> Utils::CreateDefaultBuffer(const void* initData,	UINT64 byteSize)

{
	DirectX::ResourceUploadBatch resourceUpload(Graphics::s_device->Get());
	resourceUpload.Begin(D3D12_COMMAND_LIST_TYPE_COPY);

	DirectX::SharedGraphicsResource buffer;
	ComPtr<ID3D12Resource> staticBuffer;

	buffer = Graphics::Renderer::s_graphicsMemory->Allocate(byteSize);
	memcpy(buffer.Memory(), initData, byteSize);

	CD3DX12_HEAP_PROPERTIES defaultHeap(D3D12_HEAP_TYPE_DEFAULT);

	auto desc = CD3DX12_RESOURCE_DESC::Buffer(buffer.Size());

	ThrowIfFailed(Graphics::s_device->Get()->CreateCommittedResource(
		&defaultHeap,
		D3D12_HEAP_FLAG_NONE,
		&desc,
		D3D12_RESOURCE_STATE_COMMON,
		nullptr,
		IID_PPV_ARGS(staticBuffer.GetAddressOf())));

	resourceUpload.Upload(staticBuffer.Get(), buffer);

	resourceUpload.Transition(staticBuffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);

	buffer.Reset();

	auto finish = resourceUpload.End(Graphics::s_commandQueueManager->GetCopyQueue().Get());

	finish.wait();

	return staticBuffer;
}

std::wstring Utils::GetFileName(const std::wstring& filepath)
{
	namespace fs = std::filesystem;

	auto fsfilepath = fs::path(filepath);

	assert(fs::exists(fsfilepath) && "File not found");

	std::wstring filename = fsfilepath.filename().wstring();

	return filename;
}

std::wstring Utils::GetFileDirectory(const std::wstring& filepath)
{
	namespace fs = std::filesystem;

	auto fsfilepath = fs::path(filepath);

	assert(fs::exists(fsfilepath) && "File not found");

	std::wstring directoryPath = fsfilepath.parent_path().wstring();

	return directoryPath;
}

std::wstring Utils::GetWorkingDirectory()
{
	return std::filesystem::current_path().wstring();
}

void Utils::SetWorkingDirectory(const std::wstring& path)
{
	namespace fs = std::filesystem;

	auto filesystempath = fs::path(path);

	assert(fs::exists(filesystempath) && "Directory not found");
	assert(fs::is_directory(filesystempath) && "Path is not a directory");

	fs::current_path(filesystempath);
}

std::wstring Utils::StartingWorkingDirectoryPath = std::filesystem::current_path().wstring();