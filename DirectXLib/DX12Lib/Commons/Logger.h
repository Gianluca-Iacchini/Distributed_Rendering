#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace DX12Lib {

	class Logger
	{
	public:
		Logger() {}
		~Logger() {}

		static void Init();
		inline static std::shared_ptr<spdlog::logger>& GetCoreLogger() { return s_coreLogger; }
		inline static std::shared_ptr<spdlog::logger>& GetClientLogger() { return s_clientLogger; }

	private:
		static std::shared_ptr<spdlog::logger> s_coreLogger;
		static std::shared_ptr<spdlog::logger> s_clientLogger;
	};

}


	// Core log Macros
#define DXLIB_CORE_FATAL(...) DX12Lib::Logger:GetCoreLogger()->critical(__VA_ARGS__)
#define DXLIB_CORE_ERROR(...) DX12Lib::Logger::GetCoreLogger()->error(__VA_ARGS__)
#define DXLIB_CORE_WARN(...) DX12Lib::Logger::GetCoreLogger()->warn(__VA_ARGS__)
#define DXLIB_CORE_INFO(...) DX12Lib::Logger::GetCoreLogger()->info(__VA_ARGS__)
#define DXLIB_CORE_TRACE(...) DX12Lib::Logger::GetCoreLogger()->trace(__VA_ARGS__)

// Client log Macros
#define DXLIB_FATAL(...) DX12Lib::Logger::GetClientLogger()->critical(__VA_ARGS__)
#define DXLIB_ERROR(...) DX12Lib::Logger::GetClientLogger()->error(__VA_ARGS__)
#define DXLIB_WARN(...) DX12Lib::Logger::GetClientLogger()->warn(__VA_ARGS__)
#define DXLIB_INFO(...) DX12Lib::Logger::GetClientLogger()->info(__VA_ARGS__)
#define DXLIB_TRACE(...) DX12Lib::Logger::GetClientLogger()->trace(__VA_ARGS__)

