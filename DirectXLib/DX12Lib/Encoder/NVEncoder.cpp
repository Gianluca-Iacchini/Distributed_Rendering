#include "DX12Lib/pch.h"
#include "NVEncoder.h"

using namespace DX12Lib;

NV_ENCODE_API_FUNCTION_LIST NVEncoder::m_nvEncodeAPI = { 0 };

void NVEncoder::Initialize()
{
	if (m_nvEncodeAPI.version == 0)
	{
		uint32_t version = 0;
		uint32_t currentVersion = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
		NVENC_API_CALL(NvEncodeAPIGetMaxSupportedVersion(&version));
		if (currentVersion > version)
		{
			NVENC_THROW_ERROR("Current Driver Version does not support this NvEncodeAPI version, please upgrade driver", NV_ENC_ERR_INVALID_VERSION);
		}

		m_nvEncodeAPI.version = NV_ENCODE_API_FUNCTION_LIST_VER;
		NVENC_API_CALL(NvEncodeAPICreateInstance(&m_nvEncodeAPI));
	}

	IUnknown* pDevice = Graphics::s_device->Get();

	NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS encodeSessionExParams = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
	encodeSessionExParams.device = (void*)pDevice;
	encodeSessionExParams.deviceType = NV_ENC_DEVICE_TYPE_DIRECTX;
	encodeSessionExParams.apiVersion = NVENCAPI_VERSION;

	NVENC_API_CALL(m_nvEncodeAPI.nvEncOpenEncodeSessionEx(&encodeSessionExParams, &m_hEncoder));

	UINT encodeGUIDCount = 0;
	NVENC_API_CALL(m_nvEncodeAPI.nvEncGetEncodeGUIDCount(m_hEncoder, &encodeGUIDCount));

	std::vector<GUID> encodeGUIDs(encodeGUIDCount);

	NVENC_API_CALL(m_nvEncodeAPI.nvEncGetEncodeGUIDs(m_hEncoder, encodeGUIDs.data(), encodeGUIDCount, &encodeGUIDCount));

	

}


