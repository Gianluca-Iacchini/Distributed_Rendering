#include "Helpers.h"

#ifndef SHADER_H
#define SHADER_H

class Shader
{
public:
	Shader(const std::wstring& shaderFilePath, const std::string& entryPoint, const std::string& shaderTarget);
	void AddShaderMacro(const std::string& define, const std::string& value);
	void AddShaderMacro(const std::vector<D3D_SHADER_MACRO>& defines);
	void AddShaderMacro(const D3D_SHADER_MACRO* defines, size_t num);
	void AddShaderMacro(const D3D_SHADER_MACRO& define);

	void Compile();
	Microsoft::WRL::ComPtr<ID3DBlob> GetShaderByteBlob() { return m_shaderByteBlob; };

	~Shader();

private:
	std::wstring m_shaderFilePath;
	std::string m_entryPoint;
	std::string m_shaderTarget;

	Microsoft::WRL::ComPtr<ID3DBlob> m_shaderByteBlob = nullptr;
	Microsoft::WRL::ComPtr<ID3DBlob> m_shaderErrorBlob = nullptr;
	std::vector<D3D_SHADER_MACRO> m_shaderDefines;



public:
	std::string CompileErrorMessage;
	/// <summary>
	/// Input layout for this shader (only for vertex shaders)
	/// </summary>
	std::vector<D3D12_INPUT_ELEMENT_DESC> InputLayout;

public:

	Shader(Shader&&) = default;
	Shader& operator=(Shader&&) = default;

	Shader(const Shader&) = delete;
	Shader& operator=(const Shader&) = delete;
};

#endif // !SHADER_H




