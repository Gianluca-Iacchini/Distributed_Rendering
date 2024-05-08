#include "pch.h"
#include "Shader.h"


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