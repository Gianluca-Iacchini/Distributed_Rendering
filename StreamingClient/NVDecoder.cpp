#include "NVDecoder.h"
#include "cuda.h"

#define START_TIMER auto start = std::chrono::high_resolution_clock::now()

#define STOP_TIMER(print_message) int64_t elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>( \
    std::chrono::high_resolution_clock::now() - start).count(); \
	SC_LOG_INFO("[{0}]: {1} ms\n", print_message, elapsedTime)

static const char* GetVideoCodecString(cudaVideoCodec eCodec);
static const char* GetVideoChromaFormatString(cudaVideoChromaFormat eChromaFormat);
static float GetChromaHeightFactor(cudaVideoSurfaceFormat eSurfaceFormat);
static int GetChromaPlaneCount(cudaVideoSurfaceFormat eSurfaceFormat);

SC::NVDecoder::NVDecoder(CUcontext cuContext, bool useDeviceFrame, 
	cudaVideoCodec eCodec, bool lowLatency, bool deviceFramePitched, const Rect* cropRect, 
	const Dim* resizeDim, bool extractUserSeiMessage, int maxWidth, int maxHeight, unsigned int clkRate, bool forceZeroLatency)
	: 
	m_cuContext(cuContext), m_useDeviceFrame(useDeviceFrame), m_codec(eCodec), m_deviceFramePitched(deviceFramePitched),
	m_extractSeiMessage(extractUserSeiMessage), m_maxWidth(maxWidth), m_maxHeight(maxHeight), m_forceZeroLatency(forceZeroLatency)
{
	if (cropRect) m_cropRect = *cropRect;
	if (resizeDim) m_resizeDim = *resizeDim;

	NVDEC_API_CALL(cuvidCtxLockCreate(&m_ctxLock, m_cuContext));

	CUDA_SAFE_CALL(cuStreamCreate(&m_cuvidStream, CU_STREAM_DEFAULT));

	if (m_extractSeiMessage)
	{
		m_fpSei = fopen("sei_message.txt", "wb");
		m_currSeiMessage = new CUVIDSEIMESSAGEINFO;
		memset(&m_SeiMessageDisplayOrder, 0, sizeof(m_SeiMessageDisplayOrder));
	}

	CUVIDPARSERPARAMS videoParserParameters = {};
	videoParserParameters.CodecType = m_codec;
	videoParserParameters.ulMaxNumDecodeSurfaces = 1;
	videoParserParameters.ulClockRate = clkRate;
	videoParserParameters.ulMaxDisplayDelay = lowLatency ? 0 : 1;
	videoParserParameters.pUserData = this;
	videoParserParameters.pfnSequenceCallback = HandleVideoSequenceProc;
	videoParserParameters.pfnDecodePicture = HandlePictureDecodeProc;
	videoParserParameters.pfnDisplayPicture = m_forceZeroLatency ? NULL : HandlePictureDisplayProc;
	videoParserParameters.pfnGetOperatingPoint = HandleOperatingPointProc;
	videoParserParameters.pfnGetSEIMsg = m_extractSeiMessage ? HandleSeiMessageProc : NULL;
	NVDEC_API_CALL(cuvidCreateVideoParser(&m_parser, &videoParserParameters));
}

SC::NVDecoder::~NVDecoder()
{
	EndDecode();
}

void SC::NVDecoder::EndDecode()
{
	START_TIMER;

	if (m_currSeiMessage)
	{
		delete m_currSeiMessage;
		m_currSeiMessage = NULL;
	}

	if (m_fpSei)
	{
		fclose(m_fpSei);
		m_fpSei = NULL;
	}

	if (m_parser)
		cuvidDestroyVideoParser(m_parser);

	if (m_cuContext)
		cuCtxPushCurrent(m_cuContext);

	if (m_decoder)
		cuvidDestroyDecoder(m_decoder);

	std::lock_guard<std::mutex> lock(m_frameMutex);

	for (uint8_t* frame : m_videoFrames)
	{
		if (m_useDeviceFrame)
		{
			cuMemFree((CUdeviceptr)frame);
		}
		else
		{
			delete[] frame;
		}
	}

	if (m_cuContext)
		cuCtxPopCurrent(NULL);


	if (m_ctxLock)
		cuvidCtxLockDestroy(m_ctxLock);

	STOP_TIMER("NVDecoder Destructor");
}

int SC::NVDecoder::Decode(const uint8_t* pData, int size, int flags, int64_t nTimestamp)
{
	m_nDecodedFrames = 0;
	m_nDecodedFrameReturned = 0;

	CUVIDSOURCEDATAPACKET packet = { 0 };
	packet.payload = pData;
	packet.payload_size = size;
	packet.flags = flags | CUVID_PKT_TIMESTAMP;
	packet.timestamp = nTimestamp;

	if (!pData || size == 0)
	{
		packet.flags |= CUVID_PKT_ENDOFSTREAM;
	}

	NVDEC_API_CALL(cuvidParseVideoData(m_parser, &packet));

	return m_nDecodedFrames;
}

uint8_t* SC::NVDecoder::GetFrame(int64_t* timestamp)
{
	if (m_nDecodedFrames > 0)
	{
		std::lock_guard<std::mutex> lock(m_frameMutex);
		m_nDecodedFrames--;

		if (timestamp)
		{
			*timestamp = m_videoFrameTimestamps[m_nDecodedFrameReturned];
		}

		return m_videoFrames[m_nDecodedFrameReturned++];
	}

	return NULL;
}

uint8_t* SC::NVDecoder::GetLockedFrame(int64_t* pTimeStamp)
{
	uint8_t* frame;
	uint64_t timestamp;

	if (m_nDecodedFrames > 0)
	{
		std::lock_guard<std::mutex> lock(m_frameMutex);
		m_nDecodedFrames--;
		frame = m_videoFrames[0];
		m_videoFrames.erase(m_videoFrames.begin(), m_videoFrames.begin() + 1);

		timestamp = m_videoFrameTimestamps[0];
		m_videoFrameTimestamps.erase(m_videoFrameTimestamps.begin(), m_videoFrameTimestamps.begin() + 1);

		if (pTimeStamp)
			*pTimeStamp = timestamp;

		return frame;
	}

	return NULL;
}

void SC::NVDecoder::UnlockFrame(uint8_t** frame)
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	m_videoFrames.insert(m_videoFrames.end(), &frame[0], &frame[1]);

	uint64_t timestamp[2] = { 0 };
	m_videoFrameTimestamps.insert(m_videoFrameTimestamps.end(), &timestamp[0], &timestamp[1]);
}

int SC::NVDecoder::setReconfigParams(const Rect* cropRect, const Dim* resizeDim)
{
	m_reconfigExternal = true;
	m_reconfigExtPPChange = false;

	if (cropRect)
	{
		if (!((cropRect->t == m_cropRect.t) && (cropRect->l == m_cropRect.l) &&
			(cropRect->b == m_cropRect.b) && (cropRect->r == m_cropRect.r)))
		{
			m_reconfigExtPPChange = true;
			m_cropRect = *cropRect;
		}
	}

	if (resizeDim)
	{
		if (!((resizeDim->w == m_resizeDim.w) && (resizeDim->h == m_resizeDim.h)))
		{
			m_reconfigExtPPChange = true;
			m_resizeDim = *resizeDim;
		}
	}

	uint8_t* frame = NULL;

	while (!m_videoFrames.empty())
	{
		frame = m_videoFrames.back();
		m_videoFrames.pop_back();

		if (m_useDeviceFrame)
		{
			CUDA_SAFE_CALL(cuCtxPushCurrent(m_cuContext));
			CUDA_SAFE_CALL(cuMemFree((CUdeviceptr)frame));
			CUDA_SAFE_CALL(cuCtxPopCurrent(NULL));
		}
		else
		{
			delete[] frame;
		}
	}

	return 1;
}

const char* SC::NVDecoder::GetCodecString(cudaVideoCodec eCodec)
{
	return GetVideoCodecString(eCodec);
}

int SC::NVDecoder::HandleVideoSequence(CUVIDEOFORMAT* pFormat)
{
	START_TIMER;

	m_videoInfo.str("");
	m_videoInfo.clear();

	m_videoInfo << "Video Input Information" << std::endl
		<< "\tCodec        : " << GetCodecString(pFormat->codec) << std::endl
		<< "\tFrame Rate   : " << pFormat->frame_rate.numerator << "/" << pFormat->frame_rate.denominator
		<< " = " << 1.0 * pFormat->frame_rate.numerator / pFormat->frame_rate.denominator << " fps" << std::endl
		<< "\tSequence	   :" << (pFormat->progressive_sequence ? "Progressive" : "Interlaced") << std::endl
		<< "\tCoded size   : [" << pFormat->coded_width << ", " << pFormat->coded_height << "]" << std::endl
		<< "\tDisplay Area : [" << pFormat->display_area.left << ", " << pFormat->display_area.top << ", "
		<< pFormat->display_area.right << ", " << pFormat->display_area.bottom << "]" << std::endl
		<< "\tChroma	   : " << GetVideoChromaFormatString(pFormat->chroma_format) << std::endl
		<< "\tBit Depth    : " << (pFormat->bit_depth_luma_minus8 + 8) << std::endl;

	int decodeSurface = pFormat->min_num_decode_surfaces;

	CUVIDDECODECAPS decodeCaps;
	memset(&decodeCaps, 0, sizeof(CUVIDDECODECAPS));

	decodeCaps.eCodecType = pFormat->codec;
	decodeCaps.eChromaFormat = pFormat->chroma_format;
	decodeCaps.nBitDepthMinus8 = pFormat->bit_depth_luma_minus8;

	CUDA_SAFE_CALL(cuCtxPushCurrent(m_cuContext));
	NVDEC_API_CALL(cuvidGetDecoderCaps(&decodeCaps));
	CUDA_SAFE_CALL(cuCtxPopCurrent(NULL));

	if (!decodeCaps.bIsSupported)
	{
		NVDEC_THROW_ERROR("Codec not supported", CUDA_ERROR_NOT_SUPPORTED);
	}

	if ((pFormat->coded_width > decodeCaps.nMaxWidth) ||
		(pFormat->coded_height > decodeCaps.nMaxHeight))
	{
		SC_LOG_ERROR("Resolution {0}x{1} not supported by decoder (max {2}x{3})", pFormat->coded_width, pFormat->coded_height, decodeCaps.nMaxWidth, decodeCaps.nMaxHeight);
		NVDEC_THROW_ERROR("Resolution not supported", CUDA_ERROR_NOT_SUPPORTED);
	}

	if ((pFormat->coded_width >> 4) * (pFormat->coded_height >> 4) > decodeCaps.nMaxMBCount)
	{
		SC_LOG_ERROR("MBCount {0} not supported by decoder (max {1})", (pFormat->coded_width >> 4) * (pFormat->coded_height >> 4), decodeCaps.nMaxMBCount);
		NVDEC_THROW_ERROR("MBCount not supported", CUDA_ERROR_NOT_SUPPORTED);
	}

	if (m_width && m_lumaHeight && m_chromaHeight)
		return ReconfigureDecoder(pFormat);

	m_codec = pFormat->codec;
	m_chromaFormat = pFormat->chroma_format;
	m_bitDepthMinus8 = pFormat->bit_depth_luma_minus8;
	m_bitsPerPixel = (m_bitDepthMinus8 > 0) ? 2 : 1;

	if (m_chromaFormat == cudaVideoChromaFormat_420 || m_chromaFormat == cudaVideoChromaFormat_Monochrome)
		m_outputFormat = pFormat->bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
	else if (m_chromaFormat == cudaVideoChromaFormat_444)
		m_outputFormat = pFormat->bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_YUV444_16Bit : cudaVideoSurfaceFormat_YUV444;
	else if (m_chromaFormat == cudaVideoChromaFormat_422)
		m_outputFormat = cudaVideoSurfaceFormat_NV12;

	if (!(decodeCaps.nOutputFormatMask & (1 << m_outputFormat)))
	{
		if (decodeCaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_NV12))
		{
			m_outputFormat = cudaVideoSurfaceFormat_NV12;
			SC_LOG_WARN("Chroma format not supported. Using NV12");
		}
		else if (decodeCaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_YUV444))
		{
			m_outputFormat = cudaVideoSurfaceFormat_YUV444;
			SC_LOG_WARN("Chroma format not supported. Using YUV444");
		}
		else if (decodeCaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_P016))
		{
			m_outputFormat = cudaVideoSurfaceFormat_P016;
			SC_LOG_WARN("Chroma format not supported. Using P016");
		}
		else
		{
			SC_LOG_ERROR("No supported output format found");
			NVDEC_THROW_ERROR("No supported output format found", CUDA_ERROR_NOT_SUPPORTED);
		}
	}

	m_videoFormat = *pFormat;

	CUVIDDECODECREATEINFO videoDecodeCreateInfo = { 0 };
	videoDecodeCreateInfo.CodecType = pFormat->codec;
	videoDecodeCreateInfo.ChromaFormat = pFormat->chroma_format;
	videoDecodeCreateInfo.OutputFormat = m_outputFormat;
	videoDecodeCreateInfo.bitDepthMinus8 = pFormat->bit_depth_luma_minus8;
	if (pFormat->progressive_sequence)
		videoDecodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
	else
		videoDecodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;
	videoDecodeCreateInfo.ulNumOutputSurfaces = 2;
	videoDecodeCreateInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
	videoDecodeCreateInfo.ulNumDecodeSurfaces = decodeSurface;
	videoDecodeCreateInfo.vidLock = m_ctxLock;
	videoDecodeCreateInfo.ulWidth = pFormat->coded_width;
	videoDecodeCreateInfo.ulHeight = pFormat->coded_height;

	if (m_maxWidth < (int)pFormat->coded_width)
		m_maxWidth = pFormat->coded_width;
	if (m_maxHeight < (int)pFormat->coded_height)
		m_maxHeight = pFormat->coded_height;

	videoDecodeCreateInfo.ulMaxWidth = m_maxWidth;
	videoDecodeCreateInfo.ulMaxHeight = m_maxHeight;

	if (!(m_cropRect.r && m_cropRect.b) && !(m_resizeDim.w && m_resizeDim.h))
	{
		m_width = pFormat->display_area.right - pFormat->display_area.left;
		m_lumaHeight = pFormat->display_area.bottom - pFormat->display_area.top;
		videoDecodeCreateInfo.ulTargetWidth = pFormat->coded_width;
		videoDecodeCreateInfo.ulTargetHeight = pFormat->coded_height;
	}
	else
	{
		if (m_resizeDim.w && m_resizeDim.h)
		{
			videoDecodeCreateInfo.display_area.left = pFormat->display_area.left;
			videoDecodeCreateInfo.display_area.right = pFormat->display_area.right;
			videoDecodeCreateInfo.display_area.top = pFormat->display_area.top;
			videoDecodeCreateInfo.display_area.bottom = pFormat->display_area.bottom;
			m_width = m_resizeDim.w;
			m_lumaHeight = m_resizeDim.h;
		}

		if (m_cropRect.r && m_cropRect.b)
		{
			videoDecodeCreateInfo.display_area.left = m_cropRect.l;
			videoDecodeCreateInfo.display_area.right = m_cropRect.r;
			videoDecodeCreateInfo.display_area.top = m_cropRect.t;
			videoDecodeCreateInfo.display_area.bottom = m_cropRect.b;
			m_width = m_cropRect.r - m_cropRect.l;
			m_lumaHeight = m_cropRect.b - m_cropRect.t;
		}

		videoDecodeCreateInfo.ulTargetWidth = m_width;
		videoDecodeCreateInfo.ulTargetHeight = m_lumaHeight;
	}

	m_chromaHeight = (int)(ceil(m_lumaHeight * GetChromaHeightFactor(m_outputFormat)));
	m_numChromaPlanes = GetChromaPlaneCount(m_outputFormat);
	m_surfaceHeight = videoDecodeCreateInfo.ulTargetHeight;
	m_surfaceWidth = videoDecodeCreateInfo.ulTargetWidth;
	m_displayRect.b = videoDecodeCreateInfo.display_area.bottom;
	m_displayRect.l = videoDecodeCreateInfo.display_area.left;
	m_displayRect.r = videoDecodeCreateInfo.display_area.right;
	m_displayRect.t = videoDecodeCreateInfo.display_area.top;

	m_videoInfo << "Video Decoding Params:" << std::endl
		<< "\tNum Surfaces	: " << videoDecodeCreateInfo.ulNumDecodeSurfaces << std::endl
		<< "\tCrop Rect		: [" << videoDecodeCreateInfo.display_area.left << ", " << videoDecodeCreateInfo.display_area.top << ", "
		<< videoDecodeCreateInfo.display_area.right << ", " << videoDecodeCreateInfo.display_area.bottom << "]" << std::endl
		<< "\tResize Dim	: " << videoDecodeCreateInfo.ulTargetWidth << "x" << videoDecodeCreateInfo.ulTargetHeight << std::endl
		<< "\tDeinterlace	: " << std::vector<const char*> {"Weave", "Bob", "Adaptive" }[videoDecodeCreateInfo.DeinterlaceMode] << std::endl;

	CUDA_SAFE_CALL(cuCtxPushCurrent(m_cuContext));
	NVDEC_API_CALL(cuvidCreateDecoder(&m_decoder, &videoDecodeCreateInfo));
	CUDA_SAFE_CALL(cuCtxPopCurrent(NULL));

	STOP_TIMER("Session Initialization");
	return decodeSurface;
}

int SC::NVDecoder::HandlePictureDisplay(CUVIDPARSERDISPINFO* pPicParams)
{
	CUVIDPROCPARAMS videoProcessingParameters = {};
	videoProcessingParameters.progressive_frame = pPicParams->progressive_frame;
	videoProcessingParameters.second_field = pPicParams->repeat_first_field + 1;
	videoProcessingParameters.top_field_first = pPicParams->top_field_first;
	videoProcessingParameters.unpaired_field = (pPicParams->repeat_first_field < 0);
	videoProcessingParameters.output_stream = m_cuvidStream;

	if (m_extractSeiMessage)
	{
		if (m_SeiMessageDisplayOrder[pPicParams->picture_index].pSEIData)
		{
			uint8_t* seiBuffer = (uint8_t*)(m_SeiMessageDisplayOrder[pPicParams->picture_index].pSEIData);
			uint32_t seiNumMessages = m_SeiMessageDisplayOrder[pPicParams->picture_index].sei_message_count;
			CUSEIMESSAGE* seiMessagesInfo = m_SeiMessageDisplayOrder[pPicParams->picture_index].pSEIMessage;

			if (m_fpSei)
			{
				for (uint32_t i = 0; i < seiNumMessages; i++)
				{
					if ((m_codec == cudaVideoCodec_H264) ||
						(m_codec == cudaVideoCodec_H264_SVC) ||
						(m_codec == cudaVideoCodec_H264_MVC) ||
						(m_codec == cudaVideoCodec_HEVC))
					{
						switch (seiMessagesInfo[i].sei_message_type)
						{
							case SEI_TYPE_TIME_CODE:
							case SEI_TYPE_TIME_CODE_H264:
							{
								TIMECODE* timecode = (TIMECODE*)seiBuffer;
								fwrite(timecode, sizeof(TIMECODE), 1, m_fpSei);
							}
							break;
							case SEI_TYPE_USER_DATA_REGISTERED:
							case SEI_TYPE_USER_DATA_UNREGISTERED:
							{
								fwrite(seiBuffer, seiMessagesInfo[i].sei_message_size, 1, m_fpSei);
							}
							break;
							case SEI_TYPE_MASTERING_DISPLAY_COLOR_VOLUME:
							{
								SEIMASTERINGDISPLAYINFO* masterDisplayVolume = (SEIMASTERINGDISPLAYINFO*)seiBuffer;
								fwrite(masterDisplayVolume, sizeof(SEIMASTERINGDISPLAYINFO), 1, m_fpSei);
							}
							break;
							case SEI_TYPE_CONTENT_LIGHT_LEVEL_INFO:
							{
								SEICONTENTLIGHTLEVELINFO* contentLightLevel = (SEICONTENTLIGHTLEVELINFO*)seiBuffer;
								fwrite(contentLightLevel, sizeof(SEICONTENTLIGHTLEVELINFO), 1, m_fpSei);
							}
							case SEI_TYPE_ALTERNATIVE_TRANSFER_CHARACTERISTICS:
							{
								SEIALTERNATIVETRANSFERCHARACTERISTICS* alTransferChar = (SEIALTERNATIVETRANSFERCHARACTERISTICS*)seiBuffer;
								fwrite(alTransferChar, sizeof(SEIALTERNATIVETRANSFERCHARACTERISTICS), 1, m_fpSei);
							}
							break;
						}
					}
					else
					{
						NVDEC_THROW_ERROR("Unsupported codec for SEI message extraction", CUDA_ERROR_NOT_SUPPORTED);
					}
					seiBuffer += seiMessagesInfo[i].sei_message_size;
				}
			}

			free(m_SeiMessageDisplayOrder[pPicParams->picture_index].pSEIData);
			free(m_SeiMessageDisplayOrder[pPicParams->picture_index].pSEIMessage);
		}
	}

	CUdeviceptr dpSrcFrame = 0;
	unsigned int nSrcPitch = 0;
	CUDA_SAFE_CALL(cuCtxPushCurrent(m_cuContext));
	NVDEC_API_CALL(cuvidMapVideoFrame(m_decoder, pPicParams->picture_index, &dpSrcFrame, &nSrcPitch, &videoProcessingParameters));

	CUVIDGETDECODESTATUS decodeStatus;
	memset(&decodeStatus, 0, sizeof(CUVIDGETDECODESTATUS));
	CUresult result = cuvidGetDecodeStatus(m_decoder, pPicParams->picture_index, &decodeStatus);
	if (result == CUDA_SUCCESS && (decodeStatus.decodeStatus == cuvidDecodeStatus_Error || decodeStatus.decodeStatus == cuvidDecodeStatus_Error_Concealed))
	{
		SC_LOG_ERROR("Decode Error {0} on picIndex {1}", (unsigned int)decodeStatus.decodeStatus, (unsigned int)pPicParams->picture_index);
		NVDEC_THROW_ERROR("Decode Error", CUDA_ERROR_UNKNOWN);
	}

	uint8_t* decodeFrame = nullptr;

	{
		std::lock_guard<std::mutex> lock(m_frameMutex);
		if ((unsigned)++m_nDecodedFrames > m_videoFrames.size())
		{

			m_nFrameAlloc++;
			uint8_t* frame = nullptr;
			if (m_useDeviceFrame)
			{
				if (m_deviceFramePitched)
				{
					CUDA_SAFE_CALL(cuMemAllocPitch((CUdeviceptr*)&frame, &m_nDeviceFramePitch, GetWidth() * m_bitsPerPixel, m_lumaHeight + (m_chromaHeight * m_numChromaPlanes), 16));
				}
				else
				{
					CUDA_SAFE_CALL(cuMemAlloc((CUdeviceptr*)&frame, GetFrameSize()));
				}
			}
			else
			{
				frame = new uint8_t[GetFrameSize()];
			}

			m_videoFrames.push_back(frame);
		}

		decodeFrame = m_videoFrames[m_nDecodedFrames - 1];
	}

	CUDA_MEMCPY2D m = { 0 };
	m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	m.srcDevice = dpSrcFrame;
	m.srcPitch = nSrcPitch;
	m.dstMemoryType = m_useDeviceFrame ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
	m.dstDevice = (CUdeviceptr)(m.dstHost = decodeFrame);
	m.dstPitch = m_nDeviceFramePitch ? m_nDeviceFramePitch : GetWidth() * m_bitsPerPixel;
	m.WidthInBytes = GetWidth() * m_bitsPerPixel;
	m.Height = m_lumaHeight;
	CUDA_SAFE_CALL(cuMemcpy2DAsync(&m, m_cuvidStream));

	m.srcDevice = (CUdeviceptr)((uint8_t*)dpSrcFrame + m.srcPitch * ((m_surfaceHeight + 1) & ~1));
	m.dstDevice = (CUdeviceptr)(m.dstHost = decodeFrame + m.dstPitch * m_lumaHeight);
	m.Height = m_chromaHeight;
	CUDA_SAFE_CALL(cuMemcpy2DAsync(&m, m_cuvidStream));

	if (m_numChromaPlanes == 2)
	{
		m.srcDevice = (CUdeviceptr)((uint8_t*)dpSrcFrame + m.srcPitch * ((m_surfaceHeight + 1) & ~1) * 2);
		m.dstDevice = (CUdeviceptr)(m.dstHost = decodeFrame + m.dstPitch * m_lumaHeight * 2);
		m.Height = m_chromaHeight;
		CUDA_SAFE_CALL(cuMemcpy2DAsync(&m, m_cuvidStream));
	}

	CUDA_SAFE_CALL(cuStreamSynchronize(m_cuvidStream));
	CUDA_SAFE_CALL(cuCtxPopCurrent(NULL));

	if ((int)m_videoFrameTimestamps.size() < m_nDecodedFrames)
	{
		m_videoFrameTimestamps.resize(m_videoFrames.size());
	}
	m_videoFrameTimestamps[m_nDecodedFrames - 1] = pPicParams->timestamp;

	NVDEC_API_CALL(cuvidUnmapVideoFrame(m_decoder, dpSrcFrame));

	return 1;
}

int SC::NVDecoder::HandlePictureDecode(CUVIDPICPARAMS* pPicParams)
{
	if (!m_decoder)
	{
		NVDEC_THROW_ERROR("Decoder not initialized", CUDA_ERROR_NOT_INITIALIZED);
	}

	m_nPicNumInDeocdeOrder[pPicParams->CurrPicIdx] = m_nDecodePicCnt++;
	CUDA_SAFE_CALL(cuCtxPushCurrent(m_cuContext));
	NVDEC_API_CALL(cuvidDecodePicture(m_decoder, pPicParams));

	if (m_forceZeroLatency && ((!pPicParams->field_pic_flag) || (pPicParams->second_field)))
	{
		CUVIDPARSERDISPINFO dispInfo;
		memset(&dispInfo, 0, sizeof(CUVIDPARSERDISPINFO));
		dispInfo.picture_index = pPicParams->CurrPicIdx;
		dispInfo.progressive_frame = !pPicParams->field_pic_flag;
		dispInfo.top_field_first = pPicParams->bottom_field_flag ^ 1;
		HandlePictureDisplay(&dispInfo);
	}

	CUDA_SAFE_CALL(cuCtxPopCurrent(NULL));
	return 1;
}

int SC::NVDecoder::HandleOperatingPoint(CUVIDOPERATINGPOINTINFO* pOperatingPointInfo)
{
	return -1;
}

int SC::NVDecoder::HandleSeiMessage(CUVIDSEIMESSAGEINFO* pMessage)
{
	uint32_t seiNumMessages = pMessage->sei_message_count;
	CUSEIMESSAGE* seiMessagesInfo = pMessage->pSEIMessage;
	size_t totalSeiBufferSize = 0;

	if ((pMessage->picIdx < 0) || (pMessage->picIdx >= MAX_FRAME_CNT))
	{
		NVDEC_THROW_ERROR("Invalid picture index", CUDA_ERROR_INVALID_VALUE);
	}

	for (uint32_t i = 0; i < seiNumMessages; i++)
	{
		totalSeiBufferSize += seiMessagesInfo[i].sei_message_size;
	}

	if (!m_currSeiMessage)
	{
		NVDEC_THROW_ERROR("Out of memory, Allocation failed for SEI m_currSeiMessage", CUDA_ERROR_OUT_OF_MEMORY);
	}

	m_currSeiMessage->pSEIData = malloc(totalSeiBufferSize);

	if (!m_currSeiMessage->pSEIData)
	{
		NVDEC_THROW_ERROR("Out of memory, Allocation failed for SEI m_currSeiMessage->pSEIData", CUDA_ERROR_OUT_OF_MEMORY);
	}

	memcpy(m_currSeiMessage->pSEIData, pMessage->pSEIData, totalSeiBufferSize);
	m_currSeiMessage->pSEIMessage = (CUSEIMESSAGE*)malloc(sizeof(CUSEIMESSAGE) * seiNumMessages);

	if (!m_currSeiMessage->pSEIMessage)
	{
		free(m_currSeiMessage->pSEIData);
		m_currSeiMessage->pSEIData = NULL;
		return 0;
	}

	memcpy(m_currSeiMessage->pSEIMessage, pMessage->pSEIMessage, sizeof(CUSEIMESSAGE) * seiNumMessages);
	m_currSeiMessage->sei_message_count = pMessage->sei_message_count;
	m_SeiMessageDisplayOrder[pMessage->picIdx] = *m_currSeiMessage;

	return 1;
}

int SC::NVDecoder::ReconfigureDecoder(CUVIDEOFORMAT* videoFormat)
{
	if (videoFormat->bit_depth_luma_minus8 != m_videoFormat.bit_depth_luma_minus8 || videoFormat->bit_depth_chroma_minus8 != m_videoFormat.bit_depth_chroma_minus8)
	{
		NVDEC_THROW_ERROR("Bit depth change not supported", CUDA_ERROR_NOT_SUPPORTED);
	}

	if (videoFormat->chroma_format != m_videoFormat.chroma_format)
	{
		NVDEC_THROW_ERROR("Chroma format change not supported", CUDA_ERROR_NOT_SUPPORTED);
	}

	bool decodeResChange = !(videoFormat->coded_width == m_videoFormat.coded_width && videoFormat->coded_height == m_videoFormat.coded_height);
	bool displayRectChange = !(videoFormat->display_area.left == m_videoFormat.display_area.left && videoFormat->display_area.right == m_videoFormat.display_area.right &&
		videoFormat->display_area.top == m_videoFormat.display_area.top && videoFormat->display_area.bottom == m_videoFormat.display_area.bottom);

	int nDecodeSurface = videoFormat->min_num_decode_surfaces;

	if ((videoFormat->coded_width > m_maxWidth) || (videoFormat->coded_height > m_maxHeight))
	{
		if (!m_reconfigExternal)
		{
			SC_LOG_ERROR("Resolution {0}x{1} not supported by decoder (max {2}x{3})", videoFormat->coded_width, videoFormat->coded_height, m_maxWidth, m_maxHeight);
			NVDEC_THROW_ERROR("Resolution not supported", CUDA_ERROR_NOT_SUPPORTED);
		}

		return 1;
	}

	if (!decodeResChange && !m_reconfigExtPPChange)
	{
		if (displayRectChange)
		{
			m_width = videoFormat->display_area.right - videoFormat->display_area.left;
			m_lumaHeight = videoFormat->display_area.bottom - videoFormat->display_area.top;
			m_chromaHeight = (int)(ceil(m_lumaHeight * GetChromaHeightFactor(m_outputFormat)));
			m_numChromaPlanes = GetChromaPlaneCount(m_outputFormat);
		}

		return 1;
	}

	CUVIDRECONFIGUREDECODERINFO reconfigParams = { 0 };

	reconfigParams.ulWidth = m_videoFormat.coded_width = videoFormat->coded_width;
	reconfigParams.ulHeight = m_videoFormat.coded_height = videoFormat->coded_height;

	reconfigParams.display_area.bottom = m_displayRect.b;
	reconfigParams.display_area.left = m_displayRect.l;
	reconfigParams.display_area.right = m_displayRect.r;
	reconfigParams.display_area.top = m_displayRect.t;
	reconfigParams.ulTargetWidth = m_surfaceWidth;
	reconfigParams.ulTargetHeight = m_surfaceHeight;

	if ((m_reconfigExternal && decodeResChange) || m_reconfigExtPPChange)
	{
		m_reconfigExternal = false;
		m_reconfigExtPPChange = false;
		m_videoFormat = *videoFormat;

		if (!(m_cropRect.r && m_cropRect.b) && !(m_resizeDim.w && m_resizeDim.h))
		{
			m_width = videoFormat->display_area.right - videoFormat->display_area.left;
			m_lumaHeight = videoFormat->display_area.bottom - videoFormat->display_area.top;
			reconfigParams.ulTargetWidth = videoFormat->coded_width;
			reconfigParams.ulTargetHeight = videoFormat->coded_height;
		}
		else
		{
			if (m_resizeDim.w && m_resizeDim.h)
			{
				reconfigParams.display_area.left = videoFormat->display_area.left;
				reconfigParams.display_area.right = videoFormat->display_area.right;
				reconfigParams.display_area.top = videoFormat->display_area.top;
				reconfigParams.display_area.bottom = videoFormat->display_area.bottom;
				m_width = m_resizeDim.w;
				m_lumaHeight = m_resizeDim.h;
			}

			if (m_cropRect.r && m_cropRect.b)
			{
				reconfigParams.display_area.left = m_cropRect.l;
				reconfigParams.display_area.right = m_cropRect.r;
				reconfigParams.display_area.top = m_cropRect.t;
				reconfigParams.display_area.bottom = m_cropRect.b;
				m_width = m_cropRect.r - m_cropRect.l;
				m_lumaHeight = m_cropRect.b - m_cropRect.t;
			}

			reconfigParams.ulTargetWidth = m_width;
			reconfigParams.ulTargetHeight = m_lumaHeight;
		}

		m_chromaHeight = (int)(ceil(m_lumaHeight * GetChromaHeightFactor(m_outputFormat)));
		m_numChromaPlanes = GetChromaPlaneCount(m_outputFormat);
		m_surfaceHeight = reconfigParams.ulTargetHeight;
		m_surfaceWidth = reconfigParams.ulTargetWidth;
		m_displayRect.b = reconfigParams.display_area.bottom;
		m_displayRect.l = reconfigParams.display_area.left;
		m_displayRect.r = reconfigParams.display_area.right;
		m_displayRect.t = reconfigParams.display_area.top;

	}

	reconfigParams.ulNumDecodeSurfaces = nDecodeSurface;

	START_TIMER;
	CUDA_SAFE_CALL(cuCtxPushCurrent(m_cuContext));
	NVDEC_API_CALL(cuvidReconfigureDecoder(m_decoder, &reconfigParams));
	CUDA_SAFE_CALL(cuCtxPopCurrent(NULL));
	STOP_TIMER("Reconfigure Decoder");

	return nDecodeSurface;
}


static const char* GetVideoCodecString(cudaVideoCodec eCodec) {
	static struct {
		cudaVideoCodec eCodec;
		const char* name;
	} aCodecName[] = {
		{ cudaVideoCodec_MPEG1,     "MPEG-1"       },
		{ cudaVideoCodec_MPEG2,     "MPEG-2"       },
		{ cudaVideoCodec_MPEG4,     "MPEG-4 (ASP)" },
		{ cudaVideoCodec_VC1,       "VC-1/WMV"     },
		{ cudaVideoCodec_H264,      "AVC/H.264"    },
		{ cudaVideoCodec_JPEG,      "M-JPEG"       },
		{ cudaVideoCodec_H264_SVC,  "H.264/SVC"    },
		{ cudaVideoCodec_H264_MVC,  "H.264/MVC"    },
		{ cudaVideoCodec_HEVC,      "H.265/HEVC"   },
		{ cudaVideoCodec_VP8,       "VP8"          },
		{ cudaVideoCodec_VP9,       "VP9"          },
		{ cudaVideoCodec_AV1,       "AV1"          },
		{ cudaVideoCodec_NumCodecs, "Invalid"      },
		{ cudaVideoCodec_YUV420,    "YUV  4:2:0"   },
		{ cudaVideoCodec_YV12,      "YV12 4:2:0"   },
		{ cudaVideoCodec_NV12,      "NV12 4:2:0"   },
		{ cudaVideoCodec_YUYV,      "YUYV 4:2:2"   },
		{ cudaVideoCodec_UYVY,      "UYVY 4:2:2"   },
	};

	if (eCodec >= 0 && eCodec <= cudaVideoCodec_NumCodecs) {
		return aCodecName[eCodec].name;
	}
	for (int i = cudaVideoCodec_NumCodecs + 1; i < sizeof(aCodecName) / sizeof(aCodecName[0]); i++) {
		if (eCodec == aCodecName[i].eCodec) {
			return aCodecName[eCodec].name;
		}
	}
	return "Unknown";
}

static const char* GetVideoChromaFormatString(cudaVideoChromaFormat eChromaFormat) {
	static struct {
		cudaVideoChromaFormat eChromaFormat;
		const char* name;
	} aChromaFormatName[] = {
		{ cudaVideoChromaFormat_Monochrome, "YUV 400 (Monochrome)" },
		{ cudaVideoChromaFormat_420,        "YUV 420"              },
		{ cudaVideoChromaFormat_422,        "YUV 422"              },
		{ cudaVideoChromaFormat_444,        "YUV 444"              },
	};

	if (eChromaFormat >= 0 && eChromaFormat < sizeof(aChromaFormatName) / sizeof(aChromaFormatName[0])) {
		return aChromaFormatName[eChromaFormat].name;
	}
	return "Unknown";
}

static float GetChromaHeightFactor(cudaVideoSurfaceFormat eSurfaceFormat)
{
	float factor = 0.5;
	switch (eSurfaceFormat)
	{
	case cudaVideoSurfaceFormat_NV12:
	case cudaVideoSurfaceFormat_P016:
		factor = 0.5;
		break;
	case cudaVideoSurfaceFormat_YUV444:
	case cudaVideoSurfaceFormat_YUV444_16Bit:
		factor = 1.0;
		break;
	}

	return factor;
}

static int GetChromaPlaneCount(cudaVideoSurfaceFormat eSurfaceFormat)
{
	int numPlane = 1;
	switch (eSurfaceFormat)
	{
	case cudaVideoSurfaceFormat_NV12:
	case cudaVideoSurfaceFormat_P016:
		numPlane = 1;
		break;
	case cudaVideoSurfaceFormat_YUV444:
	case cudaVideoSurfaceFormat_YUV444_16Bit:
		numPlane = 2;
		break;
	}

	return numPlane;
}