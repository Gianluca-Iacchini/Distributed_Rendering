#include "DX12Lib/pch.h"
#include "DX12Lib/Encoder/FFmpegStreamer.h"
#include "LIScene.h"
#include "DX12Lib/Scene/LightComponent.h"


using namespace LI;
using namespace DX12Lib;
using namespace Graphics;

LI::LIScene::LIScene(bool shouldStream) : DX12Lib::Scene(), m_isStreaming(shouldStream)
{
	if (m_isStreaming)
		m_ffmpegStreamer = std::make_unique<FFmpegStreamer>();
}

void LI::LIScene::Init(DX12Lib::CommandContext& context)
{
	Scene::Init(context);

	auto lightNode = this->AddNode();
	lightNode->SetPosition(0, 100, 0);
	auto light = lightNode->AddComponent<LightComponent>();
	light->SetCastsShadows(true);
	light->SetLightColor({ 0.6f, 0.6f, 0.6f });
	lightNode->Rotate(lightNode->GetRight(), 1.2f);

	if (m_isStreaming)
	{
		m_ffmpegStreamer->OpenStream(Renderer::s_clientWidth, Renderer::s_clientHeight);
		m_ffmpegStreamer->StartStreaming();
	}

}

void LI::LIScene::Update(DX12Lib::CommandContext& context)
{
	Scene::Update(context);

	if (m_isStreaming)
	{
		auto data = m_ffmpegStreamer->ConsumeData();
		this->SetNetworkData(std::get<char*>(data), std::get<size_t>(data));
	}
}

void LI::LIScene::Render(DX12Lib::CommandContext& context)
{

	Scene::Render(context);

	if (!m_isStreaming)
		return;

	// Accumulator is used to ensure proper frame rate for the encoder

	float totTime = GameTime::GetTotalTime();
	float encodeDeltaTime = totTime - m_lastUpdateTime;
	m_lastUpdateTime = totTime;
	m_accumulatedTime += encodeDeltaTime;

	float encoderFramerate = 1.f / m_ffmpegStreamer->GetEncoder().maxFrames;

	auto& backBuffer = Renderer::GetCurrentBackBuffer();

	if (m_accumulatedTime >= (encoderFramerate))
	{
		m_accumulatedTime -= encoderFramerate;
		m_ffmpegStreamer->Encode(context, backBuffer);
		int a = 0;
	}

}

void LI::LIScene::OnResize(DX12Lib::CommandContext& context, int width, int height)
{
	Scene::OnResize(context, width, height);
}

void LI::LIScene::OnClose(DX12Lib::CommandContext& context)
{
	Scene::OnClose(context);

	if (m_isStreaming)
		m_ffmpegStreamer->CloseStream();
}
