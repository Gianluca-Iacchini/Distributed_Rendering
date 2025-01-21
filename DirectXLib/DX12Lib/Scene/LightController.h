#pragma once

namespace DX12Lib
{

	class LightController : public Component
	{
	public:
		LightController() : Component() {}
		virtual ~LightController() = default;

		void Init(DX12Lib::CommandContext& context) override;
		void Update(DX12Lib::CommandContext& context) override;

	private:
		void Move(float speed, float deltaTime);

	};
}


