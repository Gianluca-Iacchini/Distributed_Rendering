#pragma once
#include "Helpers.h"
#include "assimp/scene.h"
#include "DX12Lib/DescriptorHeap.h"
#include "Texture.h"

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

	void LoadTexture(const std::wstring filename);
	void LoadTexture(const aiScene* scene);

	std::vector<std::shared_ptr<Mesh>> GetMeshes() const { return m_meshes; }

	void Draw(ID3D12GraphicsCommandList* commandList);
	void Draw(CommandList& commandList);
	void Draw(CommandContext& context);

	std::wstring ModelFolder;

private:
	void BuildVertexBuffer(UINT stride, UINT numVertices);
	void BuildIndexBuffer(DXGI_FORMAT format, UINT numIndices);

protected:
	std::vector<std::shared_ptr<Mesh>> m_meshes;

	Microsoft::WRL::ComPtr<ID3D12Resource> m_vertexBufferResource;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_indexBufferResource;

	D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;
	D3D12_INDEX_BUFFER_VIEW m_indexBufferView;

	DescriptorHeap m_textureHeap;

	std::vector<std::pair<SharedTexture, DescriptorHandle>> m_materials;
};

