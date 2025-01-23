#pragma once
#include "DX12Lib/Scene/Scene.h"

namespace DX12Lib
{
	class FFmpegStreamer;
	class CameraController;
}

namespace LI
{
	class LIScene : public DX12Lib::Scene
	{
	public:
		LIScene();
		virtual ~LIScene() = default;

		virtual void Init(DX12Lib::GraphicsContext& context) override;
		virtual void Update(DX12Lib::GraphicsContext& context) override;
		virtual void Render(DX12Lib::GraphicsContext& context) override;
		virtual void OnResize(DX12Lib::GraphicsContext& context, int width, int height) override;
		virtual void OnClose(DX12Lib::GraphicsContext& context) override;

		DX12Lib::LightComponent* GetMainLight() { return m_mainLight; }

		DX12Lib::CameraController* GetCameraController() { return m_cameraController; }

	private:
		DX12Lib::LightComponent* m_mainLight = nullptr;
		DX12Lib::CameraController* m_cameraController = nullptr;

		std::vector<char> m_inputData;
	};


}


