#pragma once
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "cuda.h"
#include "nvrtc.h"


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

}

// Log Macros

#define SC_LOG_ERROR(...) SC::Logger::GetLogger()->error(__VA_ARGS__)
#define SC_LOG_FATAL(...) SC::Logger:GetLogger()->critical(__VA_ARGS__)
#define SC_LOG_WARN(...) SC::Logger::GetLogger()->warn(__VA_ARGS__)
#define SC_LOG_INFO(...) SC::Logger::GetLogger()->info(__VA_ARGS__)
#define SC_LOG_TRACE(...) SC::Logger::GetLogger()->trace(__VA_ARGS__)
		 
extern void CUDA_SAFE_CALL(CUresult result);
		


			
													