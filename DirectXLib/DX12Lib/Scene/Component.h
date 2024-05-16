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

		virtual void Init() = 0;
		virtual void Update() = 0;
		virtual void Render() = 0;

	public:
		CommandContext* Context = nullptr;
		SceneNode* Node = nullptr;
	};
}



