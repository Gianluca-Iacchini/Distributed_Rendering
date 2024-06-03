#pragma once
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "cuda.h"
#include "nvrtc.h"
#include <memory>
#include "ColorSpace.h"

namespace SC
{
	class Helpers
	{
	};

	class Logger
	{
	public:
		Logger() {}
		~Logger() {}

		static void Init();
		inline static std::shared_ptr<spdlog::logger>& GetLogger() { return s_logger; }

	private:
		static std::shared_ptr<spdlog::logger> s_logger;
	};

	class StopWatch {
	public:
		void Start() {
			t0 = std::chrono::high_resolution_clock::now();
		}
		double Stop() {
			return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch() - t0.time_since_epoch()).count() / 1.0e9;
		}

	private:
		std::chrono::high_resolution_clock::time_point t0;
	};
}

template <class COLOR32>
void Nv12ToColor32(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix = 0);
template <class COLOR64>
void Nv12ToColor64(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix = 0);

template <class COLOR32>
void P016ToColor32(uint8_t* dpP016, int nP016Pitch, uint8_t* dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix = 4);
template <class COLOR64>
void P016ToColor64(uint8_t* dpP016, int nP016Pitch, uint8_t* dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix = 4);

template <class COLOR32>
void YUV444ToColor32(uint8_t* dpYUV444, int nPitch, uint8_t* dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix = 0);
template <class COLOR64>
void YUV444ToColor64(uint8_t* dpYUV444, int nPitch, uint8_t* dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix = 0);

template <class COLOR32>
void YUV444P16ToColor32(uint8_t* dpYUV444, int nPitch, uint8_t* dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix = 4);
template <class COLOR64>
void YUV444P16ToColor64(uint8_t* dpYUV444, int nPitch, uint8_t* dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix = 4);

template <class COLOR32>
void Nv12ToColorPlanar(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix = 0);
template <class COLOR32>
void P016ToColorPlanar(uint8_t* dpP016, int nP016Pitch, uint8_t* dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix = 4);

template <class COLOR32>
void YUV444ToColorPlanar(uint8_t* dpYUV444, int nPitch, uint8_t* dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix = 0);
template <class COLOR32>
void YUV444P16ToColorPlanar(uint8_t* dpYUV444, int nPitch, uint8_t* dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix = 4);

void Bgra64ToP016(uint8_t* dpBgra, int nBgraPitch, uint8_t* dpP016, int nP016Pitch, int nWidth, int nHeight, int iMatrix = 4);

void ConvertUInt8ToUInt16(uint8_t* dpUInt8, uint16_t* dpUInt16, int nSrcPitch, int nDestPitch, int nWidth, int nHeight);
void ConvertUInt16ToUInt8(uint16_t* dpUInt16, uint8_t* dpUInt8, int nSrcPitch, int nDestPitch, int nWidth, int nHeight);

void ResizeNv12(unsigned char* dpDstNv12, int nDstPitch, int nDstWidth, int nDstHeight, unsigned char* dpSrcNv12, int nSrcPitch, int nSrcWidth, int nSrcHeight, unsigned char* dpDstNv12UV = nullptr);
void ResizeP016(unsigned char* dpDstP016, int nDstPitch, int nDstWidth, int nDstHeight, unsigned char* dpSrcP016, int nSrcPitch, int nSrcWidth, int nSrcHeight, unsigned char* dpDstP016UV = nullptr);

void ScaleYUV420(unsigned char* dpDstY, unsigned char* dpDstU, unsigned char* dpDstV, int nDstPitch, int nDstChromaPitch, int nDstWidth, int nDstHeight,
	unsigned char* dpSrcY, unsigned char* dpSrcU, unsigned char* dpSrcV, int nSrcPitch, int nSrcChromaPitch, int nSrcWidth, int nSrcHeight, bool bSemiplanar);

#ifdef __cuda_cuda_h__
void ComputeCRC(uint8_t* pBuffer, uint32_t* crcValue, CUstream_st* outputCUStream);
#endif

// Log Macros

#define SC_LOG_ERROR(...) SC::Logger::GetLogger()->error(__VA_ARGS__)
#define SC_LOG_FATAL(...) SC::Logger:GetLogger()->critical(__VA_ARGS__)
#define SC_LOG_WARN(...) SC::Logger::GetLogger()->warn(__VA_ARGS__)
#define SC_LOG_INFO(...) SC::Logger::GetLogger()->info(__VA_ARGS__)
#define SC_LOG_TRACE(...) SC::Logger::GetLogger()->trace(__VA_ARGS__)
		 
extern void CUDA_SAFE_CALL(CUresult result);
		



													