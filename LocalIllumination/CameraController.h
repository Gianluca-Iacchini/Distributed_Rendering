#include "DX12Lib/Scene/Component.h"
#include "Keyboard.h"

namespace LI
{
	class CameraController : public DX12Lib::Component
	{
	public:
		CameraController(bool isStreaming) : Component(), m_isStreaming(isStreaming) {}

		void Init(DX12Lib::CommandContext& context) override;
		void Update(DX12Lib::CommandContext& context) override;

	private:
		void MoveUsingInput(float deltaTime, DirectX::Keyboard::State kbState);
		void MoveUsingNetwork(float deltaTime);
		void ParseInputString(std::string& input, float* mouseX, float* mouseY, int* cameraForward, int* cameraStrafe, int* cameraLift);

	private:
		bool m_isStreaming = false;
		int m_cameraForward = 0;
		int m_cameraStrafe = 0;
		int m_cameraLift = 0;

		float m_oldMouseX = 0.0f;
		float m_oldMouseY = 0.0f;

		UINT m_inputCounter = 0;
	};
}