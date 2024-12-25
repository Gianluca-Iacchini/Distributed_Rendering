#pragma once

namespace DX12Lib
{
	enum class RemoteInput
	{
		NONE = -1,
		OFF = 0,
		ON = 1,
	};

	struct NodeInputState
	{
	public:
		RemoteInput NodeForward = RemoteInput::OFF;
		RemoteInput NodeBackward = RemoteInput::OFF;
		RemoteInput NodeLeft = RemoteInput::OFF;
		RemoteInput NodeRight = RemoteInput::OFF;
		RemoteInput NodeUp = RemoteInput::OFF;
		RemoteInput NodeDown = RemoteInput::OFF;

	public:
		NodeInputState() : NodeInputState(0, 0, 0, 0, 0, 0) {}

		NodeInputState(int nodeForward, int nodeBackward, int nodeLeft, int nodeRight, int nodeUp, int nodeDown)
		{
			this->NodeForward = static_cast<RemoteInput>(nodeForward);
			this->NodeBackward = static_cast<RemoteInput>(nodeBackward);
			this->NodeLeft = static_cast<RemoteInput>(nodeLeft);
			this->NodeRight = static_cast<RemoteInput>(nodeRight);
			this->NodeUp = static_cast<RemoteInput>(nodeUp);
			this->NodeDown = static_cast<RemoteInput>(nodeDown);
		}

		NodeInputState(RemoteInput nodeForward, RemoteInput nodeBackward, RemoteInput nodeLeft, RemoteInput nodeRight, RemoteInput nodeUp, RemoteInput nodeDown)
		{
			this->NodeForward = nodeForward;
			this->NodeBackward = nodeBackward;
			this->NodeLeft = nodeLeft;
			this->NodeRight = nodeRight;
			this->NodeUp = nodeUp;
			this->NodeDown = nodeDown;
		}

		void UpdateState(NodeInputState& newState)
		{
			if (newState.NodeForward != RemoteInput::NONE)
				this->NodeForward = newState.NodeForward;
			if (newState.NodeBackward != RemoteInput::NONE)
				this->NodeBackward = newState.NodeBackward;
			if (newState.NodeLeft != RemoteInput::NONE)
				this->NodeLeft = newState.NodeLeft;
			if (newState.NodeRight != RemoteInput::NONE)
				this->NodeRight = newState.NodeRight;
			if (newState.NodeUp != RemoteInput::NONE)
				this->NodeUp = newState.NodeUp;
			if (newState.NodeDown != RemoteInput::NONE)
				this->NodeDown = newState.NodeDown;
		}

	};

	class LightController : public Component
	{
	public:
		LightController() : Component() {}
		virtual ~LightController() = default;

		void Init(DX12Lib::CommandContext& context) override;
		void Update(DX12Lib::CommandContext& context) override;

		void SetRemoteInput(NodeInputState nodeState, DirectX::XMFLOAT3 clientAbsPos, DirectX::XMFLOAT4 clientAbsRot, UINT64 clientTimestamp);
		void SetRemoteInput(std::uint8_t cameraBitmask, DirectX::XMFLOAT3 clientAbsPos, DirectX::XMFLOAT4 clientAbsRot, UINT64 timestamp);

		void ControlOverNetwork(bool control);

	private:
		void Move(float speed, float deltaTime);
		void PredictInput(float speed, float deltaTime);

		void TrimHistory(UINT64 clientTimestamp);

	private:
		struct InputDelta
		{
			DirectX::XMFLOAT3 PositionDelta;
			float deltaTime;
			UINT64 timestamp;
		};

	private:
		bool m_controlOverNetwork = false;
		NodeInputState m_inputState;

		const float m_maxPositionError = 0.5f;
		const float m_maxRotationError = 0.1f;
		const UINT c_maxHistorySize = 13;
		std::vector<InputDelta> m_inputDeltaHistory;


	};
}


