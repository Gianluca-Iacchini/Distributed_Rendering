#pragma once
#include "Component.h"
#include "GraphicsMemory.h"
#include "DX12Lib/Commons/CommonConstants.h"


namespace DX12Lib
{
	class ShadowCamera;

	class LightComponent : public Component
	{
	public:
		enum LightType
		{
			Directional,
			Point,
			Spot
		};

	public:
		LightComponent();
		~LightComponent();

		void Init(CommandContext& context) override;
		void Update(CommandContext& context) override;
		void Render(CommandContext& context) override;

		void SetLightType(LightType type) { m_lightType = type; }
		LightType GetLightType() const { return m_lightType; }

		void SetLightColor(DirectX::XMFLOAT3 color);
		DirectX::XMFLOAT3 GetLightColor() const { return m_color; }
		
		void SetLightIntensity(float intensity);
		float GetLightIntensity() const { return m_intensity; }
		
		void SetLightFalloff(float falloffStart, float falloffEnd) { m_lightCB.FalloffStart = falloffStart;  m_lightCB.FalloffEnd = falloffEnd; }
				
		bool CastsShadows() const { return m_doesCastShadows; }
		void SetCastsShadows(bool value);
		ShadowCamera* GetShadowCamera();

		ConstantBufferLight GetLightCB() const { return m_lightCB; }

		bool DidLightPropertyChange() const { return m_didLightPropertyChange; }
	public:
		static int GetLightCount() { return m_activeLights.size(); }
		static void UpdateLights(CommandContext& context);
		static void RenderLights(CommandContext& context);
		static DirectX::GraphicsResource& GetLightBufferSRV() { return m_lightBufferSRV; }

	private:
		static void RemoveLight(int index);

	private:
		UINT m_lightIndex;
		LightType m_lightType = LightType::Directional;
		ConstantBufferLight m_lightCB;
		std::unique_ptr<ShadowCamera> m_shadowCamera = nullptr;
		bool m_doesCastShadows = false;
		bool m_didLightPropertyChange = false;
		float m_intensity = 1.0f;
		DirectX::XMFLOAT3 m_color = DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f);

	private:
		static std::vector<LightComponent*> m_activeLights;
		static DirectX::GraphicsResource m_lightBufferSRV;
		static ConstantBufferLight* s_lightBufferData;
	};
}



