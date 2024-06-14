#include "DX12Lib/Scene/Component.h"

namespace LI
{
	class CameraController : public DX12Lib::Component
	{
	public:
		CameraController() : Component() {};

		void Update(DX12Lib::CommandContext& context) override;

	private:
		void ParseInputString(std::string& input, float* mouseX, float* mouseY, int* cameraForward, int* cameraStrafe, int* cameraLift);

	private:
		int m_cameraForward = 0;
		int m_cameraStrafe = 0;
		int m_cameraLift = 0;

		float m_oldMouseX = 0.0f;
		float m_oldMouseY = 0.0f;

		UINT m_inputCounter = 0;
	};
}