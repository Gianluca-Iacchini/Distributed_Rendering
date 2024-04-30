#pragma once
#include "Helpers.h"

class Mesh;
class CommandContext;
class CommandList;

class Model
{
public:
	Model() {}
	~Model() {}

	bool LoadFromFile(const std::wstring filename);
	bool LoadFromFile(const char* filename);

	std::vector<std::shared_ptr<Mesh>> GetMeshes() const { return m_meshes; }

	void Draw(ID3D12GraphicsCommandList* commandList);
	void Draw(CommandList& commandList);
	void Draw(CommandContext& context);

protected:
	std::vector<std::shared_ptr<Mesh>> m_meshes;

};

