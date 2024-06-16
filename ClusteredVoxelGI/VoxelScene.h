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

	protected:
		virtual void OnModelChildAdded(DX12Lib::SceneNode& modelNode, DX12Lib::MeshRenderer& meshRenderer, DX12Lib::ModelRenderer& modelRenderer) override;
	};
}