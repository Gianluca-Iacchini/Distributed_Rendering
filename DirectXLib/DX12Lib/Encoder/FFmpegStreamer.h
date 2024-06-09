#pragma once

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include "libavutil/error.h"
}

#include "NVEncoder.h"
#include <winsock2.h>
#include <ws2tcpip.h>
#include <tuple>


namespace DX12Lib {


	class FFmpegStreamer
	{
	public:
		FFmpegStreamer();
		~FFmpegStreamer();

		void OpenStream(UINT width, UINT height, const std::string url = "");
		void StartStreaming();
		void CloseStream();

		const NVEncoder& GetEncoder() const { return m_encoder; }
		void Encode(CommandContext& context, Resource& resource) { m_encoder.SendResourceForEncode(context, resource); }
		std::tuple<char*, size_t> ConsumeData();

	private:
		void InitWinsock();
		void StreamLoop();
		void SendFrame(std::uint8_t* data, size_t size);

	private:

		NVEncoder m_encoder;

		AVFormatContext* m_fmtCtx;
		AVStream* m_stream;
		std::string m_url;

		SOCKET m_sockfd;
		struct sockaddr_in m_servaddr, m_cliaddr;

		bool m_isStreamOpen = false;

		std::thread m_streamThread;

		std::tuple<char*, size_t> m_recvData;
	};
}

