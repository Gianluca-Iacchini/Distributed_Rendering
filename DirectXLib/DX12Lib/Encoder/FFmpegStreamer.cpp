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

	m_recvData = std::make_tuple(nullptr, 0);
}

DX12Lib::FFmpegStreamer::~FFmpegStreamer()
{

	CloseStream();

	avformat_free_context(m_fmtCtx);

}

void DX12Lib::FFmpegStreamer::OpenStream(UINT width, UINT height, std::string url)
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

	InitWinsock();

	m_encoder.StartEncodeLoop();
	m_streamThread = std::thread(&FFmpegStreamer::StreamLoop, this);
}

void DX12Lib::FFmpegStreamer::InitWinsock()
{
	WSADATA wsaData;
	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
		DXLIB_CORE_FATAL("WSAStartup failed.");
	}

	m_sockfd = socket(AF_INET, SOCK_DGRAM, 0);
	if (m_sockfd == INVALID_SOCKET) {
		WSACleanup();
		DXLIB_CORE_FATAL("Socket creation failed.");
	}

	u_long mode = 1;
	if (ioctlsocket(m_sockfd, FIONBIO, &mode) == SOCKET_ERROR) {
		closesocket(m_sockfd);
		WSACleanup();
		DXLIB_CORE_FATAL("ioctlsocket failed.");
	}

	memset(&m_servaddr, 0, sizeof(m_servaddr));
	memset(&m_cliaddr, 0, sizeof(m_cliaddr));

	m_servaddr.sin_family = AF_INET;
	m_servaddr.sin_addr.s_addr = INADDR_ANY;
	m_servaddr.sin_port = htons(12345);

	if (bind(m_sockfd, (struct sockaddr*)&m_servaddr, sizeof(m_servaddr)) == SOCKET_ERROR) {
		closesocket(m_sockfd);
		WSACleanup();
		DXLIB_CORE_FATAL("Bind failed.");
	}
}

std::tuple<char*, size_t> DX12Lib::FFmpegStreamer::ConsumeData()
{
	auto data = m_recvData;

	m_recvData = std::make_tuple(nullptr, 0);

	return data;
}

void DX12Lib::FFmpegStreamer::StreamLoop()
{
	char buffer[1024];
	int len, n, error;

	len = sizeof(m_cliaddr);


	while (m_isStreamOpen && m_encoder.IsEncoding())
	{
		std::vector<uint8_t> packets = m_encoder.ConsumePacket();
		if (!packets.empty())
		{
			this->SendFrame(packets.data(), packets.size());
		}

		n = recvfrom(m_sockfd, buffer, sizeof(buffer) - 1, 0, (struct sockaddr*)&m_cliaddr, &len);
		if (n == SOCKET_ERROR) {
			error = WSAGetLastError();
			if (error != WSAEWOULDBLOCK) {
				DXLIB_CORE_ERROR("Receive failed.");
			}

			continue;
		}

		buffer[n] = '\0';
		
		m_recvData = std::make_tuple(buffer, n);
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
		
		m_encoder.StopEncodeLoop();

		if (m_streamThread.joinable())
		{
			m_streamThread.join();
		}

		closesocket(m_sockfd);
		WSACleanup();

		FFMPEG_CHECK_ERROR(av_write_trailer(m_fmtCtx));
		FFMPEG_CHECK_ERROR(avio_close(m_fmtCtx->pb));
	}
}
