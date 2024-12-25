#pragma once
#include "DX12Lib/Scene/Scene.h"

namespace DX12Lib
{
	class FFmpegStreamer;
}

namespace LI
{
	class LIScene : public DX12Lib::Scene
	{
	public:
		LIScene(bool shouldStream);
		virtual ~LIScene() = default;

		virtual void Init(DX12Lib::CommandContext& context) override;
		virtual void Update(DX12Lib::CommandContext& context) override;
		virtual void Render(DX12Lib::CommandContext& context) override;
		virtual void OnResize(DX12Lib::CommandContext& context, int width, int height) override;
		virtual void OnClose(DX12Lib::CommandContext& context) override;

		void StreamScene(DX12Lib::CommandContext& context);
		//void TraverseModel(DX12Lib::ModelRenderer* model, aiNode* node, DX12Lib::SceneNode* parent);
		
		const std::vector<char>& GetNetworkData() const { return m_inputData; }

		DX12Lib::LightComponent* GetMainLight() { return m_mainLight; }

	private:
		void SetNetworkData(const char* data, size_t size);


	private:
		DX12Lib::LightComponent* m_mainLight = nullptr;

		bool m_isStreaming = false;
		std::unique_ptr<DX12Lib::FFmpegStreamer> m_ffmpegStreamer = nullptr;

		float m_accumulatedTime = 0;
		float m_lastUpdateTime = 0;

		std::vector<char> m_inputData;
	};


}


