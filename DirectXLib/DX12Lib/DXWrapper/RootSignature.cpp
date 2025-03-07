#include "DX12Lib/pch.h"

#include "RootSignature.h"


using namespace Microsoft::WRL;
using namespace Graphics;
using namespace DX12Lib;

void RootSignature::InitStaticSampler(UINT nRegister, const D3D12_SAMPLER_DESC& samplerDesc, D3D12_SHADER_VISIBILITY visibility)
{
	assert(m_numInitializedSamplers < m_numStaticSamplers);

	D3D12_STATIC_SAMPLER_DESC& desc = m_staticSamplers[m_numInitializedSamplers++];

	desc.Filter = samplerDesc.Filter;
	desc.AddressU = samplerDesc.AddressU;
	desc.AddressV = samplerDesc.AddressV;
	desc.AddressW = samplerDesc.AddressW;
	desc.MipLODBias = samplerDesc.MipLODBias;
	desc.MaxAnisotropy = samplerDesc.MaxAnisotropy;
	desc.ComparisonFunc = samplerDesc.ComparisonFunc;
	desc.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;
	desc.MinLOD = samplerDesc.MinLOD;
	desc.MaxLOD = samplerDesc.MaxLOD;
	desc.ShaderRegister = nRegister;
	desc.RegisterSpace = 0;
	desc.ShaderVisibility = visibility;

	if (desc.AddressU == D3D12_TEXTURE_ADDRESS_MODE_BORDER ||
		desc.AddressV == D3D12_TEXTURE_ADDRESS_MODE_BORDER ||
		desc.AddressW == D3D12_TEXTURE_ADDRESS_MODE_BORDER)
	{

		if (samplerDesc.BorderColor[3] == 1.0f)
		{
			if (samplerDesc.BorderColor[0] == 1.0f)
				desc.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;
			else if (samplerDesc.BorderColor[0] == 0.0f)
				desc.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
		}
		else
		{
			desc.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
		}
	}
}

void RootSignature::Finalize(D3D12_ROOT_SIGNATURE_FLAGS flags)
{
	if (m_finalized)
		return;

	assert(m_numStaticSamplers == m_numInitializedSamplers);

	D3D12_ROOT_SIGNATURE_DESC desc;
	desc.NumParameters = m_numParameters;
	desc.pParameters = (const D3D12_ROOT_PARAMETER*)m_rootParameters.get();
	desc.NumStaticSamplers = m_numStaticSamplers;
	desc.pStaticSamplers = (const D3D12_STATIC_SAMPLER_DESC*)m_staticSamplers.get();
	desc.Flags = flags;

	auto& param = m_rootParameters.get()[1];

	m_descirptorTableBitMap = 0;
	m_samplerTableBitMap = 0;


	for (UINT param = 0; param < m_numParameters; param++)
	{
		const D3D12_ROOT_PARAMETER& rootParam = desc.pParameters[param];
		m_descriptorTableSize[param] = 0;

		if (rootParam.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
		{
			assert(rootParam.DescriptorTable.pDescriptorRanges != nullptr);

			if (rootParam.DescriptorTable.pDescriptorRanges->RangeType == D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER)
				m_samplerTableBitMap |= (1 << param);
			else
				m_descirptorTableBitMap |= (1 << param);

			for (UINT tableRange = 0; tableRange < rootParam.DescriptorTable.NumDescriptorRanges; tableRange++)
			{
				m_descriptorTableSize[param] += rootParam.DescriptorTable.pDescriptorRanges[tableRange].NumDescriptors;
			}
		}
	}

	bool firstCompile = false;

	if (!firstCompile)
	{
		ComPtr<ID3DBlob> outBlob, errorBlob;

		HRESULT res = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, outBlob.GetAddressOf(), errorBlob.GetAddressOf());
		if (FAILED(res))
		{
			if (errorBlob)
			{
				OutputDebugStringA((char*)errorBlob->GetBufferPointer());
			}
			ThrowIfFailed(res);
		}

		ThrowIfFailed(s_device->GetComPtr()->CreateRootSignature(0, outBlob->GetBufferPointer(), outBlob->GetBufferSize(), IID_PPV_ARGS(m_rootSignature.GetAddressOf())));
	}

	m_finalized = true;
}
