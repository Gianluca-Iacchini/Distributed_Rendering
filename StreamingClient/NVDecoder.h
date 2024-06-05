#pragma once
#include "Helpers.h"
#include <nvcuvid.h>
#include <vector>
#include <mutex>
#include <sstream>

#define MAX_FRAME_CNT 32

namespace SC {

	struct Rect {
		int l, t, r, b;
	};

	struct Dim {
		int w, h;
	};

	typedef enum {
		SEI_TYPE_TIME_CODE_H264 = 1,
		SEI_TYPE_USER_DATA_REGISTERED = 4,
		SEI_TYPE_USER_DATA_UNREGISTERED = 5,
		SEI_TYPE_TIME_CODE = 136,
		SEI_TYPE_MASTERING_DISPLAY_COLOR_VOLUME = 137,
		SEI_TYPE_CONTENT_LIGHT_LEVEL_INFO = 144,
		SEI_TYPE_ALTERNATIVE_TRANSFER_CHARACTERISTICS = 147
	} SEI_H264_HEVC_MPEG2_PAYLOAD_TYPE;

	class NVDecoder
	{
	public:
		NVDecoder(CUcontext cuContext, bool useDeviceFrame, cudaVideoCodec eCodec,
			bool lowLatency = false,
			bool deviceFramePitched = false,
			const Rect* cropRect = NULL,
			const Dim* resizeDim = NULL,
			bool extractUserSeiMessage = false,
			int maxWidth = 0,
			int maxHeight = 0,
			unsigned int clkRate = 1000,
			bool forceZeroLatency = false);
		
		~NVDecoder();
		void EndDecode();

		CUcontext GetContext() const { return m_cuContext; }
		int GetWidth()
		{
			assert (m_width != 0);
			return (m_outputFormat == cudaVideoSurfaceFormat_NV12 || m_outputFormat == cudaVideoSurfaceFormat_P016) 
				? (m_width + 1) & ~1: m_width;
		}

		int Decode(const uint8_t* pData, int size, int flags = 0, int64_t nTimestamp = 0);
		
		uint8_t* GetFrame(int64_t* timestamp = nullptr);
		uint8_t* GetLockedFrame(int64_t* timestamp = nullptr);
		void UnlockFrame(uint8_t** frame);
		
		void SetOperatingPoint(const uint32_t opPoint, const bool displayAllLayers) { m_operatingPoint = opPoint; m_displayAllLayers = displayAllLayers; }
		int setReconfigParams(const Rect* cropRect, const Dim* resizeDim);
		void startTimer() { m_stDecode_time.Start(); }
		double stopTimer() { return m_stDecode_time.Stop(); }

		int GetDecodeWidth() { assert(m_width != 0); return m_width; }
		int GetHeight() { assert(m_lumaHeight != 0); return m_lumaHeight; }

		int GetChromaHeight() { assert(m_chromaHeight != 0); return m_chromaHeight; }
		int GetNumChromaPlanes() { assert(m_numChromaPlanes != 0); return m_numChromaPlanes; }
		
		int GetFrameSize() { return GetWidth() * (m_lumaHeight + (m_chromaHeight * m_numChromaPlanes)) * m_bitsPerPixel; }
		int GetLumaPlaneSize() { return GetWidth() * m_lumaHeight * m_bitsPerPixel; }
		int GetChromaPlaneSize() { return GetWidth() * (m_chromaHeight * m_numChromaPlanes) * m_bitsPerPixel; }
		int GetDeviceFramePitch() { assert(m_width); return m_nDeviceFramePitch ? (int)m_nDeviceFramePitch : GetWidth() * m_bitsPerPixel; }
		
		int GetBitDepth()  {  assert(m_width);  return m_bitDepthMinus8 + 8; }
		int GetBitsPerPixel() { assert(m_width); return m_bitsPerPixel; }
		
		cudaVideoSurfaceFormat GetOutputFormat() { return m_outputFormat; }
		CUVIDEOFORMAT GetVideoFormat() { assert(m_width); return m_videoFormat; }
		const char* GetCodecString(cudaVideoCodec eCodec);
		std::string GetVideoInfo() { return m_videoInfo.str(); }

		void ConvertFrame(uint8_t* pFrame, CUdeviceptr devPtr, int pitch);

	public:
		static int CUDAAPI HandleVideoSequenceProc(void* pUserData, CUVIDEOFORMAT* pFormat) { return ((NVDecoder*)pUserData)->HandleVideoSequence(pFormat); };
		static int CUDAAPI HandlePictureDisplayProc(void* pUserData, CUVIDPARSERDISPINFO* pPicParams) { return ((NVDecoder*)pUserData)->HandlePictureDisplay(pPicParams); };
		static int CUDAAPI HandlePictureDecodeProc(void* pUserData, CUVIDPICPARAMS* pPicParams) { return ((NVDecoder*)pUserData)->HandlePictureDecode(pPicParams); };
		static int CUDAAPI HandleOperatingPointProc(void* pUserData, CUVIDOPERATINGPOINTINFO* pOperatingPointInfo) { return ((NVDecoder*)pUserData)->HandleOperatingPoint(pOperatingPointInfo); };
		static int CUDAAPI HandleSeiMessageProc(void* pUserData, CUVIDSEIMESSAGEINFO* pMessage) { return ((NVDecoder*)pUserData)->HandleSeiMessage(pMessage); };

		int HandleVideoSequence(CUVIDEOFORMAT* pFormat);
		int HandlePictureDisplay(CUVIDPARSERDISPINFO* pPicParams);
		int HandlePictureDecode(CUVIDPICPARAMS* pPicParams);
		int HandleOperatingPoint(CUVIDOPERATINGPOINTINFO* pOperatingPointInfo);
		int HandleSeiMessage(CUVIDSEIMESSAGEINFO* pMessage);

		int ReconfigureDecoder(CUVIDEOFORMAT* videoFormat);

	private:
		CUcontext m_cuContext = NULL;
		CUvideoctxlock m_ctxLock;
		CUvideoparser m_parser = NULL;
		CUvideodecoder m_decoder = NULL;
		bool m_useDeviceFrame;

		unsigned int m_width = 0, m_lumaHeight = 0, m_chromaHeight = 0;
		unsigned int m_numChromaPlanes = 0;

		int m_surfaceHeight = 0;
		int m_surfaceWidth = 0;

		cudaVideoCodec m_codec = cudaVideoCodec_NumCodecs;
		cudaVideoChromaFormat m_chromaFormat = cudaVideoChromaFormat_420;
		cudaVideoSurfaceFormat m_outputFormat = cudaVideoSurfaceFormat_NV12;

		int m_bitDepthMinus8 = 0;
		int m_bitsPerPixel = 1;
		CUVIDEOFORMAT m_videoFormat = {};
		Rect m_displayRect = {};

		std::vector<uint8_t*> m_videoFrames;
		std::vector<int64_t> m_videoFrameTimestamps;

		int m_nDecodedFrames = 0, m_nDecodedFrameReturned = 0;
		int m_nDecodePicCnt = 0,  m_nPicNumInDeocdeOrder[MAX_FRAME_CNT];
		CUVIDSEIMESSAGEINFO* m_currSeiMessage = NULL;
		CUVIDSEIMESSAGEINFO m_SeiMessageDisplayOrder[MAX_FRAME_CNT];
		FILE* m_fpSei = NULL;
		bool m_endDecodeDone = false;
		std::mutex m_frameMutex;
		int m_nFrameAlloc = 0;
		CUstream m_cuvidStream = 0;
		bool m_deviceFramePitched = false;
		size_t m_nDeviceFramePitch = 0;
		Rect m_cropRect = {};
		Dim m_resizeDim = {};

		std::ostringstream m_videoInfo;
		unsigned int m_maxWidth = 0, m_maxHeight = 0;
		bool m_reconfigExternal = false;
		bool m_reconfigExtPPChange = false;
		StopWatch m_stDecode_time;

		unsigned int m_operatingPoint = 0;
		bool m_displayAllLayers = false;
		bool m_forceZeroLatency = false;
		bool m_extractSeiMessage = false;
	};

}



#define NVDEC_THROW_ERROR( errorStr, errorCode )																		\
    do																													\
    {																													\
		SC_LOG_ERROR("[NVDecode]: error {0} in function {1} at {2}: {3}", errorStr, __FUNCTION__, __FILE__, __LINE__);  \
		__debugbreak();																									\
    } while (0)


#define NVDEC_API_CALL( cuvidAPI )                                                                                 \
    do                                                                                                             \
    {                                                                                                              \
        CUresult errorCode = cuvidAPI;                                                                             \
        if( errorCode != CUDA_SUCCESS)                                                                             \
        {                                                                                                          \
			NVDEC_THROW_ERROR("API Call", errorCode);                                                              \
        }                                                                                                          \
    } while (0)