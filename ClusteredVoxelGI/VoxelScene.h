#include "DX12Lib/Scene/Scene.h"

namespace CVGI
{

	class VoxelScene : public DX12Lib::Scene
	{
	public:
		VoxelScene() = default;
		virtual ~VoxelScene() = default;

		virtual void Init(DX12Lib::CommandContext& context) override;
		virtual void Update(DX12Lib::CommandContext& context) override;
		virtual void Render(DX12Lib::CommandContext& context) override;
		virtual void OnResize(DX12Lib::CommandContext& context, int newWidth, int newHeight) override;
		virtual void OnClose(DX12Lib::CommandContext& context) override;

	};
}