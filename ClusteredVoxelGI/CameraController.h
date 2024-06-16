#include "DX12Lib/Scene/Component.h"

namespace CVGI
{
	class CameraController : public DX12Lib::Component
	{
	public:
		CameraController() : Component() {}

		void Init(DX12Lib::CommandContext& context) override;
		void Update(DX12Lib::CommandContext& context) override;
	};
}