#pragma once

#include "Technique.h"
#include "DX12Lib/Commons/ShadowMap.h"

namespace VOX
{
	class SceneDepthTechnique : public Technique
	{
	public:
		SceneDepthTechnique(std::shared_ptr<TechniqueData> data, bool cameraOnly=false);
		virtual ~SceneDepthTechnique() = default;

		virtual void InitializeBuffers() override;
		virtual void PerformTechnique(DX12Lib::GraphicsContext& context) override;
		void UpdateCameraMatrices();

		virtual UINT64 GetMemoryUsage() override;
	private:

		bool m_cameraOnly = false;
		DirectX::XMFLOAT3 m_lastCameraPosition = DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f);

		// Use to sample voxels from camera PoV
		DX12Lib::ShadowCamera m_depthCamera;
		DX12Lib::ShadowCamera m_offsetDepthCamera;
		// Use to sample lit voxels from light PoV
		DX12Lib::ShadowCamera m_lightCamera;

		// Used for scene depth
		DX12Lib::ConstantBufferLight m_cameraCB;
		DX12Lib::ConstantBufferLight m_offsetCameraCB;
		DX12Lib::ConstantBufferLight m_lightCB;
	};
}


