#include "DX12Lib/Scene/Component.h"
#include "DirectXMath.h"

namespace CVGI
{
	enum class RemoteInput
	{
		NONE = -1,
		OFF = 0,
		ON = 1,
	};

	class CameraController : public DX12Lib::Component
	{
	public:
		struct CameraState
		{
		public:
			RemoteInput CameraForward = RemoteInput::OFF;
			RemoteInput CameraBackward = RemoteInput::OFF;
			RemoteInput CameraLeft = RemoteInput::OFF;
			RemoteInput CameraRight = RemoteInput::OFF;
			RemoteInput CameraUp = RemoteInput::OFF;
			RemoteInput CameraDown = RemoteInput::OFF;

		public:
			CameraState() : CameraState(0, 0, 0, 0, 0, 0) {}

			CameraState(int CameraForward, int CameraBackward, int CameraLeft, int CameraRight, int CameraUp, int CameraDown)
			{
				this->CameraForward = static_cast<RemoteInput>(CameraForward);
				this->CameraBackward = static_cast<RemoteInput>(CameraBackward);
				this->CameraLeft = static_cast<RemoteInput>(CameraLeft);
				this->CameraRight = static_cast<RemoteInput>(CameraRight);
				this->CameraUp = static_cast<RemoteInput>(CameraUp);
				this->CameraDown = static_cast<RemoteInput>(CameraDown);
			}

			CameraState(RemoteInput CameraForward, RemoteInput CameraBackward, RemoteInput CameraLeft, RemoteInput CameraRight, RemoteInput CameraUp, RemoteInput CameraDown)
			{
				this->CameraForward = CameraForward;
				this->CameraBackward = CameraBackward;
				this->CameraLeft = CameraLeft;
				this->CameraRight = CameraRight;
				this->CameraUp = CameraUp;
				this->CameraDown = CameraDown;
			}

			void UpdateState(CameraState& newState)
			{
				if (newState.CameraForward != RemoteInput::NONE)
					this->CameraForward = newState.CameraForward;
				if (newState.CameraBackward != RemoteInput::NONE)
					this->CameraBackward = newState.CameraBackward;
				if (newState.CameraLeft != RemoteInput::NONE)
					this->CameraLeft = newState.CameraLeft;
				if (newState.CameraRight != RemoteInput::NONE)
					this->CameraRight = newState.CameraRight;
				if (newState.CameraUp != RemoteInput::NONE)
					this->CameraUp = newState.CameraUp;
				if (newState.CameraDown != RemoteInput::NONE)
					this->CameraDown = newState.CameraDown;
			}

		};

	private:
		struct InputDelta
		{
			DirectX::XMFLOAT3 PositionDelta;
			float deltaTime;
			UINT64 timestamp;
		};

	public:
		CameraController() : Component() {}

		void Init(DX12Lib::CommandContext& context) override;
		void Update(DX12Lib::CommandContext& context) override;

		void SetRemoteInput(CameraState cameraState, DirectX::XMFLOAT3 clientAbsPos, DirectX::XMFLOAT4 clientAbsRot, UINT64 timestamp);
		void SetRemoteInput(std::uint8_t cameraBitmask, DirectX::XMFLOAT3 clientAbsPos, DirectX::XMFLOAT4 clientAbsRot, UINT64 timestamp);

	private:
		void Move(float speed, float deltaTime);
		void PredictInput(float speed, float deltaTime);
		void TrimHistory(UINT64 clientTimestamp);

	public:
		bool IsRemote = false;

	private:
		CameraState m_cameraInputState;

		std::vector<InputDelta> m_inputDeltaHistory;

		const float m_maxPositionError = 0.5f;
		const float m_maxRotationError = 0.1f;
		const UINT c_maxHistorySize = 13;
	};
}