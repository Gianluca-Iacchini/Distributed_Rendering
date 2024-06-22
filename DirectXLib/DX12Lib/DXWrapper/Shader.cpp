#include "DX12Lib/pch.h"
#include "Shader.h"
#include <fstream>


using namespace Microsoft::WRL;
using namespace DX12Lib;

Shader::Shader(const std::wstring& shaderFilePath, const std::string& entryPoint, const std::string& shaderTarget)
	: m_shaderFilePath(shaderFilePath), m_entryPoint(entryPoint), m_shaderTarget(shaderTarget)
{

}

Shader::~Shader()
{
}

void Shader::AddShaderMacro(const D3D_SHADER_MACRO* macro, size_t num)
{
	for (size_t i = 0; i < num; i++)
	{
		m_shaderDefines.push_back(macro[i]);
	}
}

void Shader::AddShaderMacro(const D3D_SHADER_MACRO& macro)
{
	m_shaderDefines.push_back(macro);
}

void Shader::AddShaderMacro(const std::string& define, const std::string& value)
{
	D3D_SHADER_MACRO macro;
	macro.Name = define.c_str();
	macro.Definition = value.c_str();
	m_shaderDefines.push_back(macro);
}

void Shader::AddShaderMacro(const std::vector<D3D_SHADER_MACRO>& defines)
{
	m_shaderDefines.insert(m_shaderDefines.end(), defines.begin(), defines.end());
}

void Shader::Compile()
{
	UINT compileFlags = 0;

#if defined(DEBUG) || defined(_DEBUG)
	compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif


	IncludeHandler includeHandler({IncludeHandler::GetShaderDirectory()});

	HRESULT hr = S_OK;
	ComPtr<ID3DBlob> errorBlob = nullptr;
	hr = D3DCompileFromFile(m_shaderFilePath.c_str(), m_shaderDefines.data(), D3D_COMPILE_STANDARD_FILE_INCLUDE, m_entryPoint.c_str(), m_shaderTarget.c_str(), compileFlags, 0, m_shaderByteBlob.GetAddressOf(), m_shaderErrorBlob.GetAddressOf());

	if (m_shaderErrorBlob != nullptr)
	{
		CompileErrorMessage = (char*)m_shaderErrorBlob->GetBufferPointer();
		std::cout << CompileErrorMessage << std::endl;
	}

	ThrowIfFailed(hr);
}


HRESULT __stdcall DX12Lib::IncludeHandler::Open(D3D_INCLUDE_TYPE IncludeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID* ppData, UINT* pBytes)
{
	std::ifstream file;
	std::string filepath;

	for (const auto& dir : m_includeDirs) {
		std::wstring fullpath = dir + L"\\" + std::wstring(pFileName, pFileName + strlen(pFileName));
		file.open(fullpath, std::ios::binary);
		if (file.is_open()) {
			filepath = std::string(fullpath.begin(), fullpath.end());
			break;
		}
	}

	if (!file.is_open()) {
		return E_FAIL;
	}

	file.seekg(0, std::ios::end);
	size_t size = static_cast<size_t>(file.tellg());
	file.seekg(0, std::ios::beg);

	char* data = new char[size];
	file.read(data, size);
	file.close();

	*ppData = data;
	*pBytes = static_cast<UINT>(size);
	return S_OK;
}

std::wstring DX12Lib::IncludeHandler::GetShaderDirectory()
{
	std::wstring curDir = Utils::ToWstring(SOURCE_DIR) + L"/DX12Lib/DXWrapper/Shaders";
	std::replace(curDir.begin(), curDir.end(), L'/', L'\\');

	return curDir;
}
