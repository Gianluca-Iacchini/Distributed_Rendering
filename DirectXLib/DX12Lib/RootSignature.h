#include "Helpers.h"

#ifndef ROOT_SIGNATURE_H
#define ROOT_SIGNATURE_H

class Device;

class RootSignature
{
public:
	RootSignature(UINT size);
	~RootSignature();

	RootSignature(Device& device, D3D12_ROOT_SIGNATURE_DESC* rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION version) {}
	RootSignature(Device& device, D3D12_ROOT_SIGNATURE_DESC* rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION version, D3D12_STATIC_SAMPLER_DESC* samplerDesc, UINT samplerCount) {}
	RootSignature(Device& device, D3D12_ROOT_SIGNATURE_DESC* rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION version, D3D12_STATIC_SAMPLER_DESC* samplerDesc, UINT samplerCount, D3D12_ROOT_SIGNATURE_FLAGS flags) {}


private:
	Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSignature;
	D3D12_ROOT_SIGNATURE_DESC m_rootSignatureDesc;

	std::vector<CD3DX12_ROOT_PARAMETER> m_rootParameters;

public:
	ID3D12RootSignature* Get() { return m_rootSignature.Get(); }
	ID3D12RootSignature** GetAddressOf() { return m_rootSignature.GetAddressOf(); }
};

#endif // !ROOT_SIGNATURE_H



