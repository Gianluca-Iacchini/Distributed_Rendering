#pragma once
#include "Component.h"
#include "GraphicsMemory.h"
#include "DX12Lib/Commons/CommonConstants.h"

namespace DX12Lib
{
	class LightComponent : public Component
	{

	public:
		LightComponent();
		~LightComponent();

		void Update(CommandContext& context) override;
		void Render(CommandContext& context) override;

		void SetLightColor(DirectX::XMFLOAT3 color) { m_lightCB.Color = color; }
		void SetLightFalloff(float falloffStart, float falloffEnd) { m_lightCB.FalloffStart = falloffStart;  m_lightCB.FalloffEnd = falloffEnd; }
				

	public:
		static int GetLightCount() { return m_activeLights.size(); }
		static void UpdateLights(CommandContext& context);
		static void RenderLights(CommandContext& context);

	private:
		static void RemoveLight(int index);

	private:
		UINT m_lightIndex;
		ConstantBufferLight m_lightCB;

	private:
		static std::vector<LightComponent*> m_activeLights;
		static DirectX::GraphicsResource m_lightBufferSRV;
	};
}



