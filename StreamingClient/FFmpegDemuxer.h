#pragma once

extern "C" {
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

#include "cuviddec.h"
#include "Helpers.h"


#define FFMPEG_ERROR(x)																	\
{																						\
	SC_LOG_ERROR("FFMPEG Error: {0} at line {1} in file {2}", x, __LINE__, __FILE__);	\
	__debugbreak();																		\
}																						\


#define FFMPEG_CHECK(x)																		\
{																							\
	if (x < 0)																				\
	{																						\
		FFMPEG_ERROR(x);																	\
	}																						\
}																							\

namespace SC {

	class DataProvider {
	public:
		virtual ~DataProvider() {}
		virtual int GetData(uint8_t* pBuf, int nBuf) = 0;
	};

	class FFmpegDemuxer
	{
	public:
		FFmpegDemuxer() {}
		~FFmpegDemuxer();

		bool OpenStream(const char* filename, int64_t timescale = 1000);

		AVFormatContext* CreateFormatContext(const char* filename, int64_t timescale = 1000);

		bool Demux(std::uint8_t** ppData, int* nVideoBytes, int64_t* pts = NULL);

		AVCodecID GetVideoCodecID() const { return m_videoCodecID; }
		const char* GetVideoCodecName() const { return avcodec_get_name(m_videoCodecID); }
		const char* GetChromaFormat() const;
		int GetWidth() const { return m_width; }
		int GetHeight() const { return m_height; }
		int GetBitDepth() const { return m_bitDepth; }
		int GetFrameSize() const { return m_width * (m_height * m_chromaHeight) * m_bitsPerPixel; }
		int GetFramerateNumerator() const { return m_frameRateNum; }
		int GetFramerateDenominator() const { return m_frameRateDen; }

	public:
		static int ReadPacket(void* opaque, uint8_t* buf, int buf_size)
		{
			return ((DataProvider*)opaque)->GetData(buf, buf_size);
		}

		static inline cudaVideoCodec FFmpeg2NvCodecId(AVCodecID codecId)
		{
			switch (codecId)
			{
			case AV_CODEC_ID_MPEG1VIDEO: return cudaVideoCodec_MPEG1;
			case AV_CODEC_ID_MPEG2VIDEO: return cudaVideoCodec_MPEG2;
			case AV_CODEC_ID_MPEG4: return cudaVideoCodec_MPEG4;
			case AV_CODEC_ID_WMV3:
			case AV_CODEC_ID_VC1: return cudaVideoCodec_VC1;
			case AV_CODEC_ID_H264: return cudaVideoCodec_H264;
			case AV_CODEC_ID_HEVC: return cudaVideoCodec_HEVC;
			case AV_CODEC_ID_VP8: return cudaVideoCodec_VP8;
			case AV_CODEC_ID_VP9: return cudaVideoCodec_VP9;
			case AV_CODEC_ID_MJPEG: return cudaVideoCodec_JPEG;
			case AV_CODEC_ID_AV1: return cudaVideoCodec_AV1;
			default: return cudaVideoCodec_NumCodecs;
			}
		}

	private:
		AVFormatContext* m_formatCtx = NULL;
		AVIOContext* m_avInOutCtx = NULL;
		AVPacket* m_packet = NULL;
		AVPacket* m_packetFiltered = NULL;
		AVBSFContext* m_bitStreamFilterCtx = NULL;

		int m_iVideoStream;
		bool m_isHEVC, m_isH264;
		AVCodecID m_videoCodecID;
		AVPixelFormat m_chromaFormat;
		int m_width, m_height, m_bitDepth, m_bitsPerPixel, m_chromaHeight;
		double m_timeBase = 0.0;
		int64_t m_timeScale = 0;

		int m_frameRateNum = 0;
		int m_frameRateDen = 0;

		uint8_t* m_dataWidthHeader = NULL;
		unsigned int m_frameCount = 0;

		bool m_discardEarlyFrames = true;

		const char* m_filePath;
	};
}

																					