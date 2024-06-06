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

DX12Lib::FFmpegStreamer::FFmpegStreamer() : m_url("")
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

void DX12Lib::FFmpegStreamer::OpenStream(std::string url)
{
	m_url = url;

	if (m_url.empty())
	{
		m_url = "udp://localhost:1234?overrun_nonfatal=1&fifo_size=50000000";
	}


	FFMPEG_CHECK_ERROR(avformat_alloc_output_context2(&m_fmtCtx, NULL, "mpegts", m_url.c_str()));

	m_stream = avformat_new_stream(m_fmtCtx, NULL);

	if (m_stream == NULL)
	{
		FFMPEG_THROW_ERROR("Could not allocate stream");
	}

	FFMPEG_CHECK_ERROR(avio_open(&m_fmtCtx->pb, m_url.c_str(), AVIO_FLAG_WRITE));

	AVDictionary* options = NULL;


	FFMPEG_CHECK_ERROR(avformat_write_header(m_fmtCtx, &options));

	m_isStreamOpen = true;
}

void DX12Lib::FFmpegStreamer::StartStreaming(NVEncoder& encoder)
{
	if (!m_isStreamOpen)
	{
		DXLIB_CORE_ERROR("Stream is not open");
		return;
	}

	m_streamThread = std::thread(&FFmpegStreamer::StreamLoop, this, std::ref(encoder));
}

void DX12Lib::FFmpegStreamer::StreamLoop(NVEncoder& encoder)
{
	while (m_isStreamOpen && encoder.IsEncoding())
	{
		std::vector<uint8_t> packets = encoder.ConsumePacket();
		if (!packets.empty())
		{
			this->SendFrame(packets.data(), packets.size());
		}

	}
}

void DX12Lib::FFmpegStreamer::SendFrame(std::uint8_t* data, size_t size)
{
	if (size <= 0)
		return;

	AVPacket pkt;
	av_init_packet(&pkt);

	pkt.data = data;
	pkt.size = size;

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
		
		if (m_streamThread.joinable())
		{
			m_streamThread.join();
		}

		FFMPEG_CHECK_ERROR(av_write_trailer(m_fmtCtx));
		FFMPEG_CHECK_ERROR(avio_close(m_fmtCtx->pb));
	}
}
