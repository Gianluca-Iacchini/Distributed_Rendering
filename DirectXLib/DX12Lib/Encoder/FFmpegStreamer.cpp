#include "FFmpegStreamer.h"
#include "NVEncoder.h"


#define FFMPEG_THROW_ERROR(x)									\
{																\
	DXLIB_CORE_ERROR("FFMPEG Error: {0}", x);					\
	__debugbreak();												\
}															

#define FFMPEG_CHECK_ERROR(x)										\
{																	\
	if (x < 0)														\
	{																\
		char errbuf[AV_ERROR_MAX_STRING_SIZE];						\
		av_make_error_string(errbuf, AV_ERROR_MAX_STRING_SIZE, x);  \
		FFMPEG_THROW_ERROR(errbuf);									\
	}																\
}

DX12Lib::FFmpegStreamer::FFmpegStreamer() : m_url(""), m_encoder(NVEncoder())
{
	m_fmtCtx = avformat_alloc_context();

	if (!m_fmtCtx)
	{
		DXLIB_CORE_ERROR("Could not allocate format context");
		return;
	}

	av_register_all();
}

DX12Lib::FFmpegStreamer::~FFmpegStreamer()
{

	CloseStream();

	avformat_free_context(m_fmtCtx);

}

void DX12Lib::FFmpegStreamer::OpenStream(UINT width, UINT height, std::string url, AVCodecID codecId)
{
	

	m_url = url;

	if (m_url.empty())
	{
		m_url = "udp://127.0.0.1:1234?overrun_nonfatal=1&fifo_size=50000000";
	}


	FFMPEG_CHECK_ERROR(avformat_alloc_output_context2(&m_fmtCtx, NULL, "mpegts", m_url.c_str()));

	m_stream = avformat_new_stream(m_fmtCtx, NULL);

	if (m_stream == NULL)
	{
		FFMPEG_THROW_ERROR("Could not allocate stream");
	}

	if (codecId == AV_CODEC_ID_HEVC)
	{
		m_encoder.SetCodecType(NV_ENC_CODEC_HEVC_GUID);
	}
	else if (codecId == AV_CODEC_ID_H264)
	{
		m_encoder.SetCodecType(NV_ENC_CODEC_H264_GUID);
	}
	else
	{
		FFMPEG_THROW_ERROR("Unsupported codec");
	}



	AVCodecParameters* vpar = m_stream->codecpar;
	vpar->codec_id = codecId;
	vpar->codec_type = AVMEDIA_TYPE_VIDEO;
	vpar->width = 1920;
	vpar->height = 1080;

	FFMPEG_CHECK_ERROR(avio_open(&m_fmtCtx->pb, m_url.c_str(), AVIO_FLAG_WRITE));

	AVDictionary* options = NULL;


	FFMPEG_CHECK_ERROR(avformat_write_header(m_fmtCtx, &options));


	m_encoder.InitializeApp(width, height);

	m_isStreamOpen = true;
}

void DX12Lib::FFmpegStreamer::StartStreaming()
{
	if (!m_isStreamOpen)
	{
		DXLIB_CORE_ERROR("Stream is not open");
		return;
	}

	m_encoder.StartEncodeLoop();
	m_streamThread = std::thread(&FFmpegStreamer::StreamLoop, this);
}


void DX12Lib::FFmpegStreamer::StreamLoop()
{

	int nPts = 0;

	while (m_isStreamOpen && m_encoder.IsEncoding())
	{
		std::vector<uint8_t> packets = m_encoder.ConsumePacket();
		if (!packets.empty())
		{
			this->SendFrame(nPts, packets.data(), packets.size());
			nPts += 1;
		}
	}
}



void DX12Lib::FFmpegStreamer::SendFrame(int nPts, std::uint8_t* data, size_t size)
{
	if (size <= 0)
		return;

	AVPacket pkt;
	av_init_packet(&pkt);

	pkt.data = data;
	pkt.size = size;
	pkt.stream_index = m_stream->index;
	pkt.pts = av_rescale_q(nPts, AVRational{ 1, 60 }, m_stream->time_base);
	pkt.dts = pkt.pts;

	if (!memcmp(data, "\x00\x00\x00\x01\x67", 5)) {
		pkt.flags |= AV_PKT_FLAG_KEY;
	}

	int sentStatus = av_write_frame(m_fmtCtx, &pkt);

	if (sentStatus < 0)
	{
		FFMPEG_CHECK_ERROR(sentStatus);
	}
}

void DX12Lib::FFmpegStreamer::CloseStream()
{
	if (m_isStreamOpen)
	{
		m_isStreamOpen = false;
		
		m_encoder.StopEncodeLoop();

		if (m_streamThread.joinable())
		{
			m_streamThread.join();
		}

		FFMPEG_CHECK_ERROR(av_write_trailer(m_fmtCtx));
		FFMPEG_CHECK_ERROR(avio_close(m_fmtCtx->pb));
	}
}

AVCodecID DX12Lib::FFmpegStreamer::GetCodecID() const
{
	AVCodecID codecId = AVCodecID::AV_CODEC_ID_NONE;

	if (m_encoder.GetCodecGUID() == NV_ENC_CODEC_HEVC_GUID)
	{
		codecId = AVCodecID::AV_CODEC_ID_HEVC;
	}
	else if (m_encoder.GetCodecGUID() == NV_ENC_CODEC_H264_GUID)
	{
		codecId = AVCodecID::AV_CODEC_ID_H264;
	}

	return codecId;
}
