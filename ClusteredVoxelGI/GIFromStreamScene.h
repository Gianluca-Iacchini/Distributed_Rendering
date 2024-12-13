#pragma once

#include "DX12Lib/Scene/Scene.h"

class GIFromStreamScene : public DX12Lib::Scene
{
public:
	GIFromStreamScene(bool shouldStream);
	virtual ~GIFromStreamScene() = default;

	virtual void Init(DX12Lib::CommandContext& context) override;
	virtual void Update(DX12Lib::CommandContext& context) override;
	virtual void Render(DX12Lib::CommandContext& context) override;
	virtual void OnResize(DX12Lib::CommandContext& context, int width, int height) override;
	virtual void OnClose(DX12Lib::CommandContext& context) override;
};

