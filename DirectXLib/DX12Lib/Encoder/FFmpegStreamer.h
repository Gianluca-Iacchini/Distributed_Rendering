#pragma once

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include "libavutil/error.h"
}

namespace DX12Lib {

	class NVEncoder;

	class FFmpegStreamer
	{
	public:
		FFmpegStreamer();
		~FFmpegStreamer();

		void OpenStream(const std::string url = "");
		void StartStreaming(NVEncoder& encoder);
		void CloseStream();

	private:
		void StreamLoop(NVEncoder& encoder);
		void SendFrame(std::uint8_t* data, size_t size);
		AVFormatContext* m_fmtCtx;
		AVStream* m_stream;
		std::string m_url;
		bool m_isStreamOpen = false;

		std::thread m_streamThread;
	};
}

