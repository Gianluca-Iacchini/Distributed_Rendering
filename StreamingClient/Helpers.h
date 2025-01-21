#pragma once
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "cuda.h"
#include "nvrtc.h"
#include <memory>
#include "ColorSpace.h"
#include <queue>
#include <cmath>

// Log Macros

#define SC_LOG_ERROR(...) SC::Logger::GetLogger()->error(__VA_ARGS__)
#define SC_LOG_FATAL(...) SC::Logger:GetLogger()->critical(__VA_ARGS__)
#define SC_LOG_WARN(...) SC::Logger::GetLogger()->warn(__VA_ARGS__)
#define SC_LOG_INFO(...) SC::Logger::GetLogger()->info(__VA_ARGS__)
#define SC_LOG_TRACE(...) SC::Logger::GetLogger()->trace(__VA_ARGS__)

#define DXLIB_CORE_ERROR SC_LOG_ERROR
#define DXLIB_CORE_FATAL SC_LOG_FATAL
#define DXLIB_CORE_WARN SC_LOG_WARN
#define DXLIB_CORE_INFO SC_LOG_INFO
#define DXLIB_CORE_TRACE SC_LOG_TRACE

extern void CUDA_SAFE_CALL(CUresult result);

namespace SC
{
	template<typename T>
	class DoubleQueue {
	public:
		DoubleQueue() {}

		void PushInput(T element) {
			std::unique_lock<std::mutex> lock(m_mutex);
			m_inputQueue.push(element);
			m_cv.notify_one();
		}

		void PushOutput(T element) {
			std::unique_lock<std::mutex> lock(m_mutex);
			m_outputQueue.push(element);
			m_cv.notify_one();
		}

		void PopInput(T& element) {
			std::unique_lock<std::mutex> lock(m_mutex);
			m_cv.wait(lock, [this] { return !m_inputQueue.empty() || m_done; });
			
			if (m_done)
				return;
			
			element = m_inputQueue.front();
			m_inputQueue.pop();
		}

		void PopOutput(T& element) {
			std::unique_lock<std::mutex> lock(m_mutex);
			m_cv.wait(lock, [this] { return !m_outputQueue.empty() || m_done; });

			if (m_done)
				return;

			element = m_outputQueue.front();
			m_outputQueue.pop();
		}

		void SetDone() {
			std::unique_lock<std::mutex> lock(m_mutex);
			m_done = true;
			m_cv.notify_all();
		}

		unsigned int GetInputSize() {
			std::unique_lock<std::mutex> lock(m_mutex);
			return m_inputQueue.size();
		}

		unsigned int GetOutputSize() {
			std::unique_lock<std::mutex> lock(m_mutex);
			return m_outputQueue.size();
		}

	private:
		int m_maxElements = 0;
		std::queue<T> m_inputQueue;
		std::queue<T> m_outputQueue;
		std::mutex m_mutex;
		std::condition_variable m_cv;
		bool m_done = false;
	};

	class Helpers
	{
	public:
		template <typename T>
		static std::string TruncateToString(T value, int n)
		{
			T factor = std::pow(10, n);

			// Truncate the value
			T truncatedValue = std::floor(value * factor) / factor;

			// Create a buffer for the formatted string
			char buffer[50];

			// Use snprintf to format the truncated value into the buffer
			snprintf(buffer, sizeof(buffer), "%.10f", truncatedValue);

			// Convert to std::string and remove any trailing zeros
			std::string result(buffer);

			// Remove trailing zeros
			result.erase(result.find_last_not_of('0') + 1, std::string::npos);

			// If the last character is a dot, remove it
			if (result.back() == '.') {
				result.pop_back();
			}

			return result;
		}
	};

	class Logger
	{
	public:
		Logger() {}
		~Logger() {}

		static void InitializeResources();
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


		



													