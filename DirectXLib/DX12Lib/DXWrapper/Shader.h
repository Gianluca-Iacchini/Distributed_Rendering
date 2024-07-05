#pragma once

#include <d3d12.h>
#include <wrl/client.h>

namespace DX12Lib {

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


	class IncludeHandler : public ID3DInclude {
	public:
		std::vector<std::wstring> m_includeDirs;

		IncludeHandler(const std::vector<std::wstring>& includeDirs) : m_includeDirs(includeDirs)
		{
			if (DX12LIB_SHADER_DIR.empty())
			{
				std::wstring curDir = Utils::ToWstring(SOURCE_DIR) + L"/DX12Lib/DXWrapper/Shaders";
				std::replace(curDir.begin(), curDir.end(), L'/', L'\\');
				DX12LIB_SHADER_DIR = curDir;
			}

			m_includeDirs.push_back(DX12LIB_SHADER_DIR);
		}

		HRESULT __stdcall Open(D3D_INCLUDE_TYPE IncludeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID* ppData, UINT* pBytes) override;

		HRESULT __stdcall Close(LPCVOID pData) override {
			if (pData != nullptr)
				delete[] reinterpret_cast<const char*>(pData);
			
			return S_OK;
		}

	public:
		static std::wstring GetShaderDirectory(std::wstring shaderPath);
		
	private:
		static std::wstring DX12LIB_SHADER_DIR;
	};
}




