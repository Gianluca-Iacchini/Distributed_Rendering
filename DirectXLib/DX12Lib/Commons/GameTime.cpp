#include "DX12Lib/pch.h"

#include "GameTime.h"

using namespace DX12Lib;

std::unique_ptr<GameTime> GameTime::s_Instance = nullptr;

GameTime::GameTime() : m_SecondsPerCount(0.0), m_DeltaTime(-1.0), m_BaseTime(0), m_PausedTime(0), m_StopTime(0), m_PrevTime(0), m_CurrTime(0), m_Stopped(false)
{
	__int64 countsPerSec;
	QueryPerformanceFrequency((LARGE_INTEGER*)&countsPerSec);
	m_SecondsPerCount = 1.0 / static_cast<double>(countsPerSec);
}

GameTime::~GameTime()
{
}

float GameTime::TotalTime() const
{
	if (m_Stopped)
	{
		return static_cast<float>(((m_StopTime - m_PausedTime) - m_BaseTime) * m_SecondsPerCount);
	}
	else
	{
		return static_cast<float>(((m_CurrTime - m_PausedTime) - m_BaseTime) * m_SecondsPerCount);
	}
}

float GameTime::DeltaTime() const
{
	return static_cast<float>(m_DeltaTime);
}

void GameTime::Reset()
{
	__int64 currTime;
	QueryPerformanceCounter((LARGE_INTEGER*)&currTime);

	m_BaseTime = currTime;
	m_PrevTime = currTime;
	m_StopTime = 0;
	m_Stopped = false;
}

void GameTime::Start()
{
	__int64 startTime;
	QueryPerformanceCounter((LARGE_INTEGER*)&startTime);

	if (m_Stopped)
	{
		m_PausedTime += (startTime - m_StopTime);
		m_PrevTime = startTime;
		m_StopTime = 0;
		m_Stopped = false;
	}
}

void GameTime::Stop()
{
	if (!m_Stopped)
	{
		__int64 currTime;
		QueryPerformanceCounter((LARGE_INTEGER*)&currTime);

		m_StopTime = currTime;
		m_Stopped = true;
	}
}

void GameTime::Tick()
{
	if (m_Stopped)
	{
		m_DeltaTime = 0.0;
		return;
	}

	__int64 currTime;
	QueryPerformanceCounter((LARGE_INTEGER*)&currTime);
	m_CurrTime = currTime;

	m_DeltaTime = (m_CurrTime - m_PrevTime) * m_SecondsPerCount;

	m_PrevTime = m_CurrTime;

	if (m_DeltaTime < 0.0)
	{
		m_DeltaTime = 0.0;
	}
}

float GameTime::GetInstantTime() const
{
	__int64 currTime;
	QueryPerformanceCounter((LARGE_INTEGER*)&currTime);

	return static_cast<float>((currTime - m_BaseTime) * m_SecondsPerCount);
}

UINT64 DX12Lib::GameTime::GetTimeSinceEpoch()
{
	auto now = std::chrono::system_clock::now();
	
	// reutrn milliseconds since epoch
	return now.time_since_epoch() / std::chrono::microseconds(1);
}

void DX12Lib::GameTime::CreateInstance()
{
	if (s_Instance == nullptr)
	{
		s_Instance = std::make_unique<GameTime>();
	}
}

