#include "DX12Lib/pch.h"
#include "NVEncoder.h"


using namespace DX12Lib;

NV_ENCODE_API_FUNCTION_LIST NVEncoder::m_nvEncodeAPI = { 0 };

DX12Lib::NVEncoder::~NVEncoder()
{
	if (m_hEncoder)
	{
	
		ReleaseInputBuffers();
		ReleaseOutputBuffers();

		m_nvEncodeAPI.nvEncDestroyEncoder(m_hEncoder);
		m_hEncoder = nullptr;
	}
}

void NVEncoder::Initialize(UINT width, UINT height)
{
	DXLIB_CORE_WARN("TODO: Unregister InputResources");

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

	UINT presetCount = 0;
	NVENC_API_CALL(m_nvEncodeAPI.nvEncGetEncodePresetCount(m_hEncoder, m_hevcCodecGUID, &presetCount));

	if (presetCount == 0)
	{
		NVENC_THROW_ERROR("No HEVC Presets Found", NV_ENC_ERR_UNSUPPORTED_PARAM);
	}

	m_initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
	m_encodeConfig = { NV_ENC_CONFIG_VER };

	memset(&m_encodeConfig, 0, sizeof(NV_ENC_CONFIG));
	memset(&m_initializeParams, 0, sizeof(NV_ENC_INITIALIZE_PARAMS));

	m_initializeParams.version = NV_ENC_INITIALIZE_PARAMS_VER;
	m_initializeParams.encodeConfig = &m_encodeConfig;
	m_initializeParams.encodeConfig->version = NV_ENC_CONFIG_VER;

	DXLIB_CORE_WARN("TODO: Check best settings for multi-pass");
	m_initializeParams.encodeConfig->rcParams.multiPass = NV_ENC_MULTI_PASS_DISABLED;
	
	m_initializeParams.encodeGUID = m_hevcCodecGUID;
	m_initializeParams.presetGUID = m_presetGUID;
	m_initializeParams.encodeWidth = width;
	m_initializeParams.encodeHeight = height;
	m_initializeParams.darWidth = width;
	m_initializeParams.darHeight = height;
	m_initializeParams.frameRateNum = maxFrames;
	m_initializeParams.frameRateDen = 1;
	m_initializeParams.enablePTD = 1;
	m_initializeParams.reportSliceOffsets = 0;
	m_initializeParams.enableSubFrameWrite = 0;
	m_initializeParams.maxEncodeWidth = width;
	m_initializeParams.maxEncodeHeight = height;
	m_initializeParams.enableMEOnlyMode = false;
	m_initializeParams.enableOutputInVidmem = false;
	m_initializeParams.tuningInfo = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;

	m_initializeParams.enableEncodeAsync = SupportsAsyncMode(m_hevcCodecGUID);

	m_initializeParams.encodeConfig->rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
	m_initializeParams.bufferFormat = NV_ENC_BUFFER_FORMAT_ARGB;


	NV_ENC_PRESET_CONFIG presetConfig = { NV_ENC_PRESET_CONFIG_VER, 0, { NV_ENC_CONFIG_VER }};
	NVENC_API_CALL(m_nvEncodeAPI.nvEncGetEncodePresetConfigEx(m_hEncoder, m_hevcCodecGUID, m_presetGUID, NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY, &presetConfig));

	memcpy(m_initializeParams.encodeConfig, &presetConfig.presetCfg, sizeof(NV_ENC_CONFIG));

	NVENC_API_CALL(m_nvEncodeAPI.nvEncInitializeEncoder(m_hEncoder, &m_initializeParams));

	m_nEncodeBuffer = m_encodeConfig.frameIntervalP + m_encodeConfig.rcParams.lookaheadDepth + 3;
	m_completionEvents.resize(m_nEncodeBuffer, nullptr);

	for (UINT i = 0; i < m_completionEvents.size(); i++)
	{
		m_completionEvents[i] = CreateEvent(NULL, FALSE, FALSE, NULL);
	}



	ThrowIfFailed(Graphics::s_device->GetComPtr()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(m_inputFence.GetAddressOf())));
	ThrowIfFailed(Graphics::s_device->GetComPtr()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(m_outputFence.GetAddressOf())));

	m_fenceEvent = CreateEvent(NULL, FALSE, FALSE, NULL);

	AllocateInputBuffers(m_nEncodeBuffer);
	AllocateOutputBuffers(m_nEncodeBuffer);

	m_mappedInputBuffers.resize(m_nEncodeBuffer, nullptr);
	m_mappedOutputBuffers.resize(m_nEncodeBuffer, nullptr);
}

void DX12Lib::NVEncoder::EncodeFrame(CommandContext& context, Resource& resource, ENCODED_PACKET& vPacket)
{
	int buffIndex = m_iToSend % m_nEncodeBuffer;

	/*
	==============================================
	TODO: Change this to make render system copy the backbuffer into the input buffer
		  I'm pretty sure that letting the encoder do the copy will cause issues with the backbuffer
		  resource state.
	==============================================
	*/

	this->MapResource(buffIndex);

	InterlockedIncrement(&m_outputFenceValue);

	m_outputResources[buffIndex]->pOutputBuffer = m_mappedOutputBuffers[buffIndex];
	m_outputResources[buffIndex]->outputFencePoint.waitValue = m_outputFenceValue;
	m_outputResources[buffIndex]->outputFencePoint.bSignal = true;

	m_inputResources[buffIndex]->pInputBuffer = m_mappedInputBuffers[buffIndex];
	m_inputResources[buffIndex]->inputFencePoint.waitValue = m_inputFenceValue;
	m_inputResources[buffIndex]->inputFencePoint.bWait = true;

	NVENCSTATUS nvStatus = Encode(m_inputResources[buffIndex].get(), m_outputResources[buffIndex].get());

	if (nvStatus == NV_ENC_SUCCESS || nvStatus == NV_ENC_ERR_NEED_MORE_INPUT)
	{
		m_iToSend++;
	}
	else
	{
		NVENC_THROW_ERROR("Encode failed", nvStatus);
	}


}

void DX12Lib::NVEncoder::EndEncode(ENCODED_PACKET& packet)
{
	packet.clear();
	if (m_hEncoder == nullptr) return;

	DXLIB_CORE_WARN("TODO: FINISH THIS");
}

bool DX12Lib::NVEncoder::SupportsAsyncMode(GUID codecGUID)
{
	if (m_hEncoder == nullptr)
		return false;
	
	NV_ENC_CAPS_PARAM capsParam = { NV_ENC_CAPS_PARAM_VER };
	capsParam.capsToQuery = NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT;
	int32_t asyncMode = 0;
	NVENC_API_CALL(m_nvEncodeAPI.nvEncGetEncodeCaps(m_hEncoder, codecGUID, &capsParam, &asyncMode));

	return asyncMode;
}

void DX12Lib::NVEncoder::AllocateInputBuffers(UINT nInputBuffers)
{
	assert(m_hEncoder != nullptr);

	DXLIB_CORE_WARN("TODO: Change with CD3DX12_HEAP_PROPERTIES if it works.");

	UINT maxWidth = m_initializeParams.maxEncodeWidth;
	UINT maxHeight = m_initializeParams.maxEncodeHeight;

	D3D12_HEAP_PROPERTIES heapProps{};
	heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
	heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
	heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

	D3D12_RESOURCE_DESC resourceDesc{};
	resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	resourceDesc.Alignment = 0;
	resourceDesc.Width = maxWidth;
	resourceDesc.Height = maxHeight;
	resourceDesc.DepthOrArraySize = 1;
	resourceDesc.MipLevels = 1;
	resourceDesc.SampleDesc.Count = 1;
	resourceDesc.SampleDesc.Quality = 0;
	resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
	resourceDesc.Format = m_bufferFormat;

	m_inputBuffers.resize(nInputBuffers);

	for (UINT i = 0; i < nInputBuffers; i++)
	{
		ThrowIfFailed(Graphics::s_device->GetComPtr()->CreateCommittedResource(
			&heapProps,
			D3D12_HEAP_FLAG_NONE,
			&resourceDesc,
			D3D12_RESOURCE_STATE_COMMON,
			nullptr,
			IID_PPV_ARGS(m_inputBuffers[i].GetAddressOf())
		));
		std::wstring name = L"Input Buffer " + std::to_wstring(i);
		m_inputBuffers[i]->SetName(name.c_str());
	}

	// We create the buffers using Directx12 and then we register them with the encoder
	RegisterInputResources(maxWidth, maxHeight);

	for (int i = 0; i < nInputBuffers; i++)
	{
		std::unique_ptr<NV_ENC_INPUT_RESOURCE_D3D12> inpRes = std::make_unique<NV_ENC_INPUT_RESOURCE_D3D12>();
		memset(inpRes.get(), 0, sizeof(NV_ENC_INPUT_RESOURCE_D3D12));
		inpRes->inputFencePoint.pFence = m_inputFence.Get();

		m_inputResources.push_back(std::move(inpRes));
	}
}

void DX12Lib::NVEncoder::AllocateOutputBuffers(UINT nOutputbuffers)
{
	assert(m_hEncoder != nullptr);

	D3D12_HEAP_PROPERTIES heapProps{};
	heapProps.Type = D3D12_HEAP_TYPE_READBACK;
	heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
	heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

	UINT bufferSize = m_initializeParams.encodeWidth * m_initializeParams.encodeHeight * 4;

	DXLIB_CORE_INFO("Output Buffer Size before alignment: {0}", bufferSize);

	bufferSize = Utils::AlignAtBytes(bufferSize, 4);

	DXLIB_CORE_INFO("Output Buffer Size after alignment: {0}", bufferSize);

	D3D12_RESOURCE_DESC resourceDesc{};
	resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	resourceDesc.Alignment = 0;
	resourceDesc.Width = bufferSize;
	resourceDesc.Height = 1;
	resourceDesc.DepthOrArraySize = 1;
	resourceDesc.MipLevels = 1;
	resourceDesc.SampleDesc.Count = 1;
	resourceDesc.SampleDesc.Quality = 0;
	resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
	resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

	m_outputBuffers.resize(nOutputbuffers);

	for (UINT i = 0; i < nOutputbuffers; i++)
	{
		ThrowIfFailed(Graphics::s_device->GetComPtr()->CreateCommittedResource(
			&heapProps,
			D3D12_HEAP_FLAG_NONE,
			&resourceDesc,
			D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr,
			IID_PPV_ARGS(m_outputBuffers[i].GetAddressOf())
		));
	}

	RegisterOutputResources(bufferSize);

	for (UINT i = 0; i < m_outputBuffers.size(); i++)
	{
		std::unique_ptr<NV_ENC_OUTPUT_RESOURCE_D3D12> outRes = std::make_unique<NV_ENC_OUTPUT_RESOURCE_D3D12>();
		memset(outRes.get(), 0, sizeof(NV_ENC_OUTPUT_RESOURCE_D3D12));
		outRes->outputFencePoint.pFence = m_outputFence.Get();
		m_outputResources.push_back(std::move(outRes));
	}
}

void DX12Lib::NVEncoder::RegisterInputResources(UINT width, UINT height)
{
	for (UINT i = 0; i < m_inputBuffers.size(); i++)
	{
		NV_ENC_FENCE_POINT_D3D12 regResInputFence;

		memset(&regResInputFence, 0, sizeof(NV_ENC_FENCE_POINT_D3D12));
		regResInputFence.pFence = m_inputFence.Get();
		regResInputFence.waitValue = m_inputFenceValue;
		regResInputFence.bWait = true;

		InterlockedIncrement(&m_inputFenceValue);

		regResInputFence.signalValue = m_inputFenceValue;
		regResInputFence.bSignal = true;

		NV_ENC_REGISTERED_PTR registeredPtr = RegisterResource(
			m_inputBuffers[i].Get(),
			width,
			height,
			NV_ENC_BUFFER_FORMAT_ARGB,
			NV_ENC_INPUT_IMAGE,
			&regResInputFence
		);

		ID3D12Resource* pRes = m_inputBuffers[i].Get();
		D3D12_RESOURCE_DESC desc = pRes->GetDesc();
		D3D12_PLACED_SUBRESOURCE_FOOTPRINT inputUploadFootprint[2];

		Graphics::s_device->Get()->GetCopyableFootprints(&desc, 0, 1, 0, inputUploadFootprint, nullptr, nullptr, nullptr);

		NvEncInputFrame inputFrame = {};
		inputFrame.inputPtr = m_inputBuffers[i].Get();
		inputFrame.pitch = inputUploadFootprint[0].Footprint.RowPitch;
		
		m_registeredResources.push_back(registeredPtr);
		m_inputFrames.push_back(inputFrame);

		WaitForFence((ID3D12Fence*)regResInputFence.pFence, regResInputFence.signalValue);
	}
}

void DX12Lib::NVEncoder::RegisterOutputResources(UINT bfrSize)
{
	for (UINT i = 0; i < m_outputBuffers.size(); i++)
	{
		NV_ENC_REGISTERED_PTR registeredPtr = RegisterResource(
			m_outputBuffers[i].Get(),
			bfrSize,
			1,
			NV_ENC_BUFFER_FORMAT_U8,
			NV_ENC_OUTPUT_BITSTREAM
		);

		m_registeredResourcesOutputBuffers.push_back(registeredPtr);

	}
}

void DX12Lib::NVEncoder::WaitForFence(ID3D12Fence* fence, UINT64 fenceValue)
{
	if (fence->GetCompletedValue() < fenceValue)
	{
		if (fence->SetEventOnCompletion(fenceValue, m_fenceEvent) != S_OK)
		{
			NVENC_THROW_ERROR("SetEventOnCompletion failed", NV_ENC_ERR_INVALID_PARAM);
		}
		WaitForSingleObject(m_fenceEvent, INFINITE);
	}
}

NV_ENC_REGISTERED_PTR DX12Lib::NVEncoder::RegisterResource(void* buffer, int width, int height, NV_ENC_BUFFER_FORMAT bufFormat, NV_ENC_BUFFER_USAGE bufUsage, NV_ENC_FENCE_POINT_D3D12* inputFencePoint)
{
	NV_ENC_REGISTER_RESOURCE registerRes = { NV_ENC_REGISTER_RESOURCE_VER };
	registerRes.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX;
	registerRes.resourceToRegister = buffer;
	registerRes.width = width;
	registerRes.height = height;
	registerRes.pitch = 0;
	registerRes.bufferFormat = bufFormat;
	registerRes.bufferUsage = bufUsage;
	registerRes.pInputFencePoint = inputFencePoint;

	NVENC_API_CALL(m_nvEncodeAPI.nvEncRegisterResource(m_hEncoder, &registerRes));

	return registerRes.registeredResource;
}

void DX12Lib::NVEncoder::MapResource(UINT buffIndex)
{
	NV_ENC_MAP_INPUT_RESOURCE mapInputResource = { NV_ENC_MAP_INPUT_RESOURCE_VER };
	mapInputResource.registeredResource = m_registeredResources[buffIndex];
	NVENC_API_CALL(m_nvEncodeAPI.nvEncMapInputResource(m_hEncoder, &mapInputResource));
	m_mappedInputBuffers[buffIndex] = mapInputResource.mappedResource;

	NV_ENC_MAP_INPUT_RESOURCE mapInputResourceBitstreamBuffer = { NV_ENC_MAP_INPUT_RESOURCE_VER };
	mapInputResourceBitstreamBuffer.registeredResource = m_registeredResourcesOutputBuffers[buffIndex];
	NVENC_API_CALL(m_nvEncodeAPI.nvEncMapInputResource(m_hEncoder, &mapInputResourceBitstreamBuffer));
	m_mappedOutputBuffers[buffIndex] = mapInputResourceBitstreamBuffer.mappedResource;
}

NVENCSTATUS DX12Lib::NVEncoder::Encode(NV_ENC_INPUT_RESOURCE_D3D12* inputResource, NV_ENC_OUTPUT_RESOURCE_D3D12* outputResource)
{
	NV_ENC_PIC_PARAMS picParams = { NV_ENC_PIC_PARAMS_VER };
	picParams.version = NV_ENC_PIC_PARAMS_VER;
	picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
	picParams.inputBuffer = inputResource;
	picParams.bufferFmt = NV_ENC_BUFFER_FORMAT_ARGB;
	picParams.inputWidth = m_initializeParams.encodeWidth;
	picParams.inputHeight = m_initializeParams.encodeHeight;
	picParams.frameIdx = m_iToSend;
	picParams.outputBitstream = outputResource;
	picParams.completionEvent = GetCompletionEvent(m_iToSend % m_nEncodeBuffer);

	NVENCSTATUS status = m_nvEncodeAPI.nvEncEncodePicture(m_hEncoder, &picParams);

	return status;
}

void DX12Lib::NVEncoder::FlushEncoder()
{
	if (m_hEncoder == nullptr) return;

	ENCODED_PACKET vPacket;
	EndEncode(vPacket);
}

void DX12Lib::NVEncoder::ReleaseInputBuffers()
{
	if (!m_hEncoder) return;

	UnregisterInputResources();

	m_inputResources.clear();
	m_inputFrames.clear();
}

void DX12Lib::NVEncoder::ReleaseOutputBuffers()
{
	if (!m_hEncoder) return;

	UnregisterOutputResources();

	m_outputResources.clear();
}

void DX12Lib::NVEncoder::UnregisterInputResources()
{
	FlushEncoder();

	for (UINT i = 0; i < m_mappedInputBuffers.size(); i++)
	{
		if (m_mappedInputBuffers[i] != nullptr)
		{
			NVENC_API_CALL(m_nvEncodeAPI.nvEncUnmapInputResource(m_hEncoder, m_mappedInputBuffers[i]));
		}
	}
	m_mappedInputBuffers.clear();

	for (UINT i = 0; i < m_registeredResources.size(); i++)
	{
		if (m_registeredResources[i] != nullptr)
		{
			NVENC_API_CALL(m_nvEncodeAPI.nvEncUnregisterResource(m_hEncoder, m_registeredResources[i]));
		}
	}
	m_registeredResources.clear();
}

void DX12Lib::NVEncoder::UnregisterOutputResources()
{
	for (UINT i = 0; i < m_registeredResourcesOutputBuffers.size(); i++)
	{
		NV_ENC_REGISTERED_PTR registeredPtr = m_registeredResourcesOutputBuffers[i];

		if (registeredPtr != nullptr)
		{
			NVENC_API_CALL(m_nvEncodeAPI.nvEncUnregisterResource(m_hEncoder, registeredPtr));
		}
	}

	m_registeredResourcesOutputBuffers.clear();
}

NvEncInputFrame& DX12Lib::NVEncoder::GetNextInputFrame()
{
	int i = m_iToSend % m_nEncodeBuffer;
	return m_inputFrames[i];
}



