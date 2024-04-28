#pragma once
#include "Helpers.h"

class Device;

class RootParameter
{
	friend class RootSignature;
public:
	RootParameter() 
	{
		m_rootParameter.ParameterType = (D3D12_ROOT_PARAMETER_TYPE)-1;
	}

	~RootParameter() 
	{
		if (m_rootParameter.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
		{
			delete[] m_rootParameter.DescriptorTable.pDescriptorRanges;
		}

		m_rootParameter.ParameterType = (D3D12_ROOT_PARAMETER_TYPE)-1;
	}

	void InitAsConstants(UINT nRegister, UINT numDWORDS, D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL, UINT nSpace = 0)
	{
		m_rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
		m_rootParameter.Constants.ShaderRegister = nRegister;
		m_rootParameter.Constants.RegisterSpace = nSpace;
		m_rootParameter.ShaderVisibility = visibility;
		m_rootParameter.Constants.Num32BitValues = numDWORDS;
	}

	void InitAsConstantBuffer(UINT nRegister, D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL, UINT nSpace = 0)
	{
		m_rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
		m_rootParameter.Descriptor.ShaderRegister = nRegister;
		m_rootParameter.Descriptor.RegisterSpace = nSpace;
		m_rootParameter.ShaderVisibility = visibility;
	}

	void InitAsBufferSRV(UINT nRegister, D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL, UINT nSpace = 0)
	{
		m_rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
		m_rootParameter.Descriptor.ShaderRegister = nRegister;
		m_rootParameter.Descriptor.RegisterSpace = nSpace;
		m_rootParameter.ShaderVisibility = visibility;
	}

	void InitAsBufferUAV(UINT nRegister, D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL, UINT nSpace = 0)
	{
		m_rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
		m_rootParameter.Descriptor.ShaderRegister = nRegister;
		m_rootParameter.Descriptor.RegisterSpace = nSpace;
		m_rootParameter.ShaderVisibility = visibility;
	}

	void InitAsDescriptorTable(UINT rangeCount, D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
	{
		m_rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		m_rootParameter.DescriptorTable.NumDescriptorRanges = rangeCount;
		m_rootParameter.ShaderVisibility = visibility;
		m_rootParameter.DescriptorTable.pDescriptorRanges = new D3D12_DESCRIPTOR_RANGE[rangeCount];
	}

	void InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE type, UINT nRegister, UINT count, D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL, UINT nSpace = 0)
	{
		InitAsDescriptorTable(1, visibility);
		SetTableRange(0, type, nRegister, count, nSpace);
	}

	void SetTableRange(UINT rangeIndex, D3D12_DESCRIPTOR_RANGE_TYPE type, UINT nRegister, UINT count, UINT space = 0)
	{
		D3D12_DESCRIPTOR_RANGE* range = const_cast<D3D12_DESCRIPTOR_RANGE*>(m_rootParameter.DescriptorTable.pDescriptorRanges + rangeIndex);
		range->RangeType = type;
		range->NumDescriptors = count;
		range->BaseShaderRegister = nRegister;
		range->RegisterSpace = space;
		range->OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
	}

	const D3D12_ROOT_PARAMETER& operator() (void) const { return m_rootParameter; }

protected:
	D3D12_ROOT_PARAMETER m_rootParameter;
};

class RootSignature
{
public:
	RootSignature(UINT numRootParameters = 0, UINT numStaticSamplers = 0) 
		: m_finalized(false)
	{
		Reset(numRootParameters, numStaticSamplers);
	}

	~RootSignature() {}

	void Reset(UINT numRootParameters, UINT numStaticSamplers = 0)
	{
		if (numRootParameters > 0)
			m_rootParameters.reset(new RootParameter[numRootParameters]);
		else
			m_rootParameters = nullptr;
	
		m_numParameters = numRootParameters;

		if (numStaticSamplers > 0)
			m_staticSamplers.reset(new D3D12_STATIC_SAMPLER_DESC[numStaticSamplers]);
		else
			m_staticSamplers = nullptr;

		m_numStaticSamplers = numStaticSamplers;
		m_numInitializedSamplers = 0;
	}

	void InitStaticSampler(UINT nRegister, const D3D12_SAMPLER_DESC& samplerDesc, D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL);

	void Finalize(D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE);

private:
	Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSignature;
	UINT m_numInitializedSamplers = 0;
	UINT m_numParameters = 0;
	UINT m_numStaticSamplers = 0;
	D3D12_ROOT_SIGNATURE_DESC m_rootSignatureDesc;
	bool m_finalized = false;

	uint32_t m_descirptorTableBitMap = 0;
	uint32_t m_samplerTableBitMap = 0;
	uint32_t m_descriptorTableSize[16];

	std::unique_ptr<RootParameter[]> m_rootParameters;
	std::unique_ptr<D3D12_STATIC_SAMPLER_DESC[]> m_staticSamplers;

public:
	ID3D12RootSignature* Get() { return m_rootSignature.Get(); }
	ID3D12RootSignature** GetAddressOf() { return m_rootSignature.GetAddressOf(); }
	Microsoft::WRL::ComPtr<ID3D12RootSignature> GetComPtr() { return m_rootSignature; }
	
	RootParameter& operator[] (size_t i) {
		assert(i < m_numParameters);
		return m_rootParameters.get()[i]; 
	}

	RootParameter& operator[] (size_t i) const {
		assert(i < m_numParameters);
		return m_rootParameters.get()[i];
	}
};




