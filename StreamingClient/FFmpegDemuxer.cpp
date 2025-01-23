#include "FFmpegDemuxer.h"


SC::FFmpegDemuxer::~FFmpegDemuxer()
{
	if (!m_formatCtx)
		return;

	if (m_packet)
		av_packet_free(&m_packet);

	if (m_packetFiltered)
		av_packet_free(&m_packetFiltered);

	if (m_bitStreamFilterCtx)
		av_bsf_free(&m_bitStreamFilterCtx);

	avformat_close_input(&m_formatCtx);

	if (m_avInOutCtx)
	{
		av_freep(&m_avInOutCtx->buffer);
		av_freep(&m_avInOutCtx);
	}

	if (m_dataWidthHeader)
		av_freep(&m_dataWidthHeader);

	avformat_network_deinit();
}

bool SC::FFmpegDemuxer::OpenStream(const char* filename, int64_t timeScale)
{
	AVFormatContext* fmtc = CreateFormatContext(filename, timeScale);

	if (!fmtc)
	{
		SC_LOG_ERROR("AVFormatContext is NULL");
		return false;
	}

	m_packet = av_packet_alloc();
	m_packetFiltered = av_packet_alloc();

	if (!m_packet || !m_packetFiltered)
	{
		SC_LOG_ERROR("av_packet_alloc failed");
		return false;
	}

	m_formatCtx = fmtc;

	std::string long_name = "";
	if (fmtc->iformat->long_name)
		long_name = fmtc->iformat->long_name;

	std::string name = "";
	if (fmtc->iformat->name)
		name = fmtc->iformat->name;

	SC_LOG_INFO("Media format: {0} ({1})", long_name, name);

	FFMPEG_CHECK(avformat_find_stream_info(fmtc, NULL));

	m_iVideoStream = av_find_best_stream(fmtc, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
	if (m_iVideoStream < 0)
	{
		av_packet_free(&m_packet);
		av_packet_free(&m_packetFiltered);
		FFMPEG_ERROR("No video stream found");
	}

	m_videoCodecID = fmtc->streams[m_iVideoStream]->codecpar->codec_id;
	m_width = fmtc->streams[m_iVideoStream]->codecpar->width;
	m_height = fmtc->streams[m_iVideoStream]->codecpar->height;
	m_chromaFormat = (AVPixelFormat)fmtc->streams[m_iVideoStream]->codecpar->format;
	AVRational rTimeBase = fmtc->streams[m_iVideoStream]->time_base;
	m_timeBase = av_q2d(rTimeBase);
	m_timeScale = timeScale;

	switch (m_chromaFormat)
	{
	case AV_PIX_FMT_YUV420P10LE:
	case AV_PIX_FMT_GRAY10LE:
		m_bitDepth = 10;
		m_chromaHeight = (m_height + 1) >> 1; // (height + 1) / 2
		m_bitsPerPixel = 2;
		break;

	case AV_PIX_FMT_YUV420P12LE:
		m_bitDepth = 12;
		m_chromaHeight = (m_height + 1) >> 1; // (height + 1) / 2
		m_bitsPerPixel = 2;
		break;

	case AV_PIX_FMT_YUV444P10LE:
		m_bitDepth = 10;
		m_chromaHeight = m_height << 1;
		m_bitsPerPixel = 2;
		break;

	case AV_PIX_FMT_YUV444P12LE:
		m_bitDepth = 12;
		m_chromaHeight = m_height << 1;
		m_bitsPerPixel = 2;
		break;

	case AV_PIX_FMT_YUV444P:
		m_bitDepth = 8;
		m_chromaHeight = m_height << 1;
		m_bitsPerPixel = 1;
		break;

	case AV_PIX_FMT_YUV420P:
	case AV_PIX_FMT_YUVJ420P:
	case AV_PIX_FMT_YUVJ422P:	 // jpeg decoder output is subsampled to NV12 for 422/444 so treat it as 420
	case AV_PIX_FMT_YUVJ444P:	 // jpeg decoder output is subsampled to NV12 for 422/444 so treat it as 420
	case AV_PIX_FMT_GRAY8:		 // monochrome is treated as 420 with chroma filled with 0x0
		m_bitDepth = 8;
		m_chromaHeight = (m_height + 1) >> 1; // (height + 1) / 2
		m_bitsPerPixel = 1;
		break;

	default:
		SC_LOG_WARN("Chroma Format not recognized. Defaulting to 8-bit 420");
		m_bitDepth = 8;
		m_chromaHeight = (m_height + 1) >> 1; // (height + 1) / 2
		m_bitsPerPixel = 1;
	}

	m_isH264 = m_videoCodecID == AV_CODEC_ID_H264 && (
		!strcmp(fmtc->iformat->long_name, "QuickTime / MOV")
		|| !strcmp(fmtc->iformat->long_name, "FLV (Flash Video)")
		|| !strcmp(fmtc->iformat->long_name, "Matroska / WebM")
		|| !strcmp(fmtc->iformat->long_name, "Apple HTTP Live Streaming")
		|| !strcmp(fmtc->iformat->name, "mpegts")
		);
	m_isHEVC = m_videoCodecID == AV_CODEC_ID_HEVC && (
		!strcmp(fmtc->iformat->long_name, "QuickTime / MOV")
		|| !strcmp(fmtc->iformat->long_name, "FLV (Flash Video)")
		|| !strcmp(fmtc->iformat->long_name, "Matroska / WebM")
		|| !strcmp(fmtc->iformat->long_name, "raw HEVC video")
		|| !strcmp(fmtc->iformat->name, "mpegts")
		);


	if (m_isH264)
	{
		const AVBitStreamFilter* bsf = av_bsf_get_by_name("h264_mp4toannexb");
		if (!bsf)
		{
			av_packet_free(&m_packet);
			av_packet_free(&m_packetFiltered);
			FFMPEG_ERROR("h264_mp4toannexb filter not found");
		}

		FFMPEG_CHECK(av_bsf_alloc(bsf, &m_bitStreamFilterCtx));
		avcodec_parameters_copy(m_bitStreamFilterCtx->par_in, fmtc->streams[m_iVideoStream]->codecpar);
		FFMPEG_CHECK(av_bsf_init(m_bitStreamFilterCtx));
	}

	if (m_isHEVC)
	{
		const AVBitStreamFilter* bsf = av_bsf_get_by_name("hevc_mp4toannexb");
		if (!bsf)
		{
			av_packet_free(&m_packet);
			av_packet_free(&m_packetFiltered);
			FFMPEG_ERROR("hevc_mp4toannexb filter not found");
		}

		FFMPEG_CHECK(av_bsf_alloc(bsf, &m_bitStreamFilterCtx));
		avcodec_parameters_copy(m_bitStreamFilterCtx->par_in, fmtc->streams[m_iVideoStream]->codecpar);
		FFMPEG_CHECK(av_bsf_init(m_bitStreamFilterCtx));
	}

	m_frameRateNum = m_formatCtx->streams[m_iVideoStream]->r_frame_rate.num;
	m_frameRateDen = m_formatCtx->streams[m_iVideoStream]->r_frame_rate.den;

	return true;
}

AVFormatContext* SC::FFmpegDemuxer::CreateFormatContext(const char* filename, int64_t timescale)
{
	avformat_network_init();

	AVDictionary* opts = NULL;
	// Set low_delay flag
	FFMPEG_CHECK(av_dict_set(&opts, "flags", "low_delay", 0));

	// Optionally set buffer size reduction
	//av_dict_set(&opts, "fflags", "nobuffer", 0);

	//// Optionally set frame dropping
	FFMPEG_CHECK(av_dict_set(&opts, "flags2", "veryfast", 0));

	// Set frame drop option
	FFMPEG_CHECK(av_dict_set(&opts, "frame_drop", "1", 0));

	// Enable low latency
	FFMPEG_CHECK(av_dict_set(&opts, "low_latency", "1", 0));

	// Set real-time flag
	FFMPEG_CHECK(av_dict_set(&opts, "realtime", "1", 0));

	// Set buffer size
	av_dict_set(&opts, "buffer_size", "102400", 0);

	// Set maximum delay
	av_dict_set(&opts, "max_delay", "50000", 0);

	// timeout
	av_dict_set(&opts, "timeout", "5000000", 0);

	AVFormatContext* ctx = NULL;

	if (avformat_open_input(&ctx, filename, NULL, &opts) < 0)
	{
		return NULL;
	}

	return ctx;
}

bool SC::FFmpegDemuxer::Demux(std::uint8_t** data, int* nVideoBytes, int64_t* pts)
{
	assert(nVideoBytes != NULL);

	if (!m_formatCtx)
		return false;

	*nVideoBytes = 0;

	static unsigned int i = 0;



	if (m_packet->data)
		av_packet_unref(m_packet);

	int e = 0;

	while ((e = av_read_frame(m_formatCtx, m_packet)) >= 0 && m_packet->stream_index != m_iVideoStream)
	{
		av_packet_unref(m_packet);
	}

	if (e < 0)
		return false;

	if (!m_isH264 && !m_isHEVC)
	{
		FFMPEG_ERROR("Unsupported codec");
	}
	
	if (m_packetFiltered->data)
		av_packet_unref(m_packetFiltered);

	FFMPEG_CHECK(av_bsf_send_packet(m_bitStreamFilterCtx, m_packet));
	FFMPEG_CHECK(av_bsf_receive_packet(m_bitStreamFilterCtx, m_packetFiltered));
	*data = m_packetFiltered->data;
	*nVideoBytes = m_packetFiltered->size;

	if (pts)
	{
		*pts = (int64_t)(m_packetFiltered->pts * m_timeBase * m_timeScale);
	}
	
	m_frameCount++;

	return true;
}

const char* SC::FFmpegDemuxer::GetChromaFormat() const
{
	return av_get_pix_fmt_name(m_chromaFormat);
}


