#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>



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



// Core log Macros
#define DXLIB_CORE_FATAL(...) Logger:GetCoreLogger()->critical(__VA_ARGS__)
#define DXLIB_CORE_ERROR(...) Logger::GetCoreLogger()->error(__VA_ARGS__)
#define DXLIB_CORE_WARN(...) Logger::GetCoreLogger()->warn(__VA_ARGS__)
#define DXLIB_CORE_INFO(...) Logger::GetCoreLogger()->info(__VA_ARGS__)
#define DXLIB_CORE_TRACE(...) Logger::GetCoreLogger()->trace(__VA_ARGS__)

// Client log Macros
#define DXLIB_FATAL(...) Logger::GetClientLogger()->critical(__VA_ARGS__)
#define DXLIB_ERROR(...) Logger::GetClientLogger()->error(__VA_ARGS__)
#define DXLIB_WARN(...) Logger::GetClientLogger()->warn(__VA_ARGS__)
#define DXLIB_INFO(...) Logger::GetClientLogger()->info(__VA_ARGS__)
#define DXLIB_TRACE(...) Logger::GetClientLogger()->trace(__VA_ARGS__)

