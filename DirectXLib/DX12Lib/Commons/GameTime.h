#pragma once

#include "memory"

namespace DX12Lib {

	class GameTime
	{
		friend class D3DApp;

	public:
		GameTime();
		~GameTime();

		float TotalTime() const;
		float DeltaTime() const;

		void Reset();
		void Start();
		void Stop();
		void Tick();

		float GetInstantTime() const;

		static inline float GetTotalTime() { return s_Instance->TotalTime(); }
		static inline float GetDeltaTime() { return s_Instance->DeltaTime(); }

	private:
		static void CreateInstance();

	private:
		static std::unique_ptr<GameTime> s_Instance;

		double m_SecondsPerCount;
		double m_DeltaTime;

		__int64 m_BaseTime;
		__int64 m_PausedTime;
		__int64 m_StopTime;
		__int64 m_PrevTime;
		__int64 m_CurrTime;

		bool m_Stopped;
	};
}
