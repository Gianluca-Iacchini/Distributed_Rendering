#include "DX12Lib/pch.h"
#include "Logger.h"

using namespace DX12Lib;

std::shared_ptr<spdlog::logger> Logger::s_coreLogger;
std::shared_ptr<spdlog::logger> Logger::s_clientLogger;

void Logger::Init()
{
	spdlog::set_pattern("%^[%T] [%n]: %v%$");
	s_coreLogger = spdlog::stdout_color_mt("DXLIB");
	s_coreLogger->set_level(spdlog::level::trace);

	s_clientLogger = spdlog::stdout_color_mt("APP");
	s_clientLogger->set_level(spdlog::level::trace);

	s_coreLogger->info(L"Core Logger initialized");
	s_clientLogger->info(L"Client Logger initialized");
}

