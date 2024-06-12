#pragma once

#include "DX12Lib/Commons/CommandContext.h"

namespace DX12Lib
{
	class SceneNode;

	class Component
	{
	public:
		Component() {}
		virtual ~Component() {}

		virtual void Init(CommandContext& context) {}
		virtual void Update(CommandContext& context) {}
		virtual void Render(CommandContext& context) {}
		virtual void OnResize(CommandContext& context, int newWidth, int newHeight) {}

	public:
		SceneNode* Node = nullptr;
	};
}



