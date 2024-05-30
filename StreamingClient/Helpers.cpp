#include "Helpers.h"

using namespace SC;

std::shared_ptr<spdlog::logger> Logger::s_logger;


void Logger::Init()
{
	spdlog::set_pattern("%^[%T] [%n]: %v%$");

	s_logger = spdlog::stdout_color_mt("APP");
	s_logger->set_level(spdlog::level::trace);

	s_logger->info(L"Streaming Client Logger initialized");
}

void CUDA_SAFE_CALL(CUresult result)
{
	if (result != CUDA_SUCCESS)
	{
		const char* error;
		cuGetErrorString(result, &error);
		SC_LOG_ERROR("CUDA Error: {0}", error);
	}
}