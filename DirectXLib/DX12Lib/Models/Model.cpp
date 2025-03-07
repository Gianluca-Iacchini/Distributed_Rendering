#include "DX12Lib/pch.h"

#include "Model.h"
#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "Mesh.h"



using namespace DX12Lib;

using namespace Assimp;
using namespace Graphics;
using namespace Renderer;

void Model::LoadFromFile(const aiScene* scene)
{
    this->LoadMaterials(scene);
    this->LoadMeshes(scene);
}

void Model::LoadMaterials(const aiScene* scene)
{
    m_materials.resize(scene->mNumMaterials);
    std::vector<ConstantBufferMaterial> materialCBs(scene->mNumMaterials);

    for (UINT i = 0; i < scene->mNumMaterials; i++)
    {
        aiMaterial* material = scene->mMaterials[i];

        auto builder = Renderer::s_materialManager->CreateMaterialBuilder();
        m_materials[i] = builder.BuildFromAssimpMaterial(material);
    }


}

void Model::LoadMeshes(const aiScene* scene)
{
    std::vector<DirectX::VertexPositionNormalTexture> vertices;
    std::vector<UINT> indices;

    UINT totalVertices = 0;
    UINT totalIndices = 0;

    for (UINT i = 0; i < scene->mNumMeshes; i++)
    {
        auto assimpMesh = scene->mMeshes[i];

        std::shared_ptr<Mesh> newMesh = std::make_shared<Mesh>();

        for (UINT v = 0; v < assimpMesh->mNumVertices; v++)
        {
            DirectX::VertexPositionNormalTexture vertex;

            if (assimpMesh->HasPositions())
                vertex.position = { assimpMesh->mVertices[v].x, assimpMesh->mVertices[v].y, assimpMesh->mVertices[v].z };

            if (assimpMesh->HasNormals())
                vertex.normal = { assimpMesh->mNormals[v].x, assimpMesh->mNormals[v].y, assimpMesh->mNormals[v].z };

            if (assimpMesh->HasTextureCoords(0))
                vertex.textureCoordinate = { assimpMesh->mTextureCoords[0][v].x, assimpMesh->mTextureCoords[0][v].y };

            m_minBounds = MathHelper::Min(m_minBounds, vertex.position);
			m_maxBounds = MathHelper::Max(m_maxBounds, vertex.position);

            vertices.push_back(vertex);
        }

        assert(assimpMesh->HasFaces() && "Mesh has no faces");

        UINT numIndices = 0;

        for (UINT j = 0; j < assimpMesh->mNumFaces; j++)
        {
            const aiFace& face = assimpMesh->mFaces[j];

            if (face.mNumIndices == 3)
            {
                indices.push_back(face.mIndices[0]);
                indices.push_back(face.mIndices[1]);
                indices.push_back(face.mIndices[2]);
            }

            numIndices += face.mNumIndices;
        }



        DXGI_FORMAT indexFormat = DXGI_FORMAT_R32_UINT;


        newMesh->m_primitiveTopology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        newMesh->m_numIndices = numIndices;
        newMesh->m_indexStart = totalIndices;
        newMesh->m_vertexStart = totalVertices;

        totalVertices += assimpMesh->mNumVertices;
        totalIndices += numIndices;

        assert(assimpMesh->mMaterialIndex < m_materials.size() && "Material index out of bounds");
        assert(m_materials[assimpMesh->mMaterialIndex] != nullptr && "Material is null");

        newMesh->MeshMaterial = m_materials[assimpMesh->mMaterialIndex].get();

        m_meshes.push_back(newMesh);
    }

    UINT vertexStride = sizeof(DirectX::VertexPositionNormalTexture);
    m_vertexBufferResource = Utils::CreateDefaultBuffer(vertices.data(), vertexStride * vertices.size());
    BuildVertexBuffer(vertexStride, vertices.size());

    m_indexBufferResource = Utils::CreateDefaultBuffer(indices.data(), sizeof(UINT) * indices.size());
    BuildIndexBuffer(DXGI_FORMAT_R32_UINT, indices.size());
}

void Model::Draw(ID3D12GraphicsCommandList* commandList)
{
    assert(commandList != nullptr && "CommandList is null");

    commandList->IASetVertexBuffers(0, 1, &m_vertexBufferView);
    commandList->IASetIndexBuffer(&m_indexBufferView);
}

void Model::Draw(CommandList* commandList)
{
    Draw(commandList->Get());
}

void Model::Draw(CommandContext& context)
{
	Draw(context.m_commandList);
}

void DX12Lib::Model::UseBuffers(CommandContext& context)
{
    context.m_commandList->Get()->IASetVertexBuffers(0, 1, &m_vertexBufferView);
    context.m_commandList->Get()->IASetIndexBuffer(&m_indexBufferView);
}

void Model::BuildVertexBuffer(UINT stride, UINT numVertices)
{
    m_vertexBufferView.BufferLocation = m_vertexBufferResource->GetGPUVirtualAddress();
    m_vertexBufferView.StrideInBytes = stride;
    m_vertexBufferView.SizeInBytes = stride * numVertices;
}

void Model::BuildIndexBuffer(DXGI_FORMAT format, UINT numIndices)
{
    m_indexBufferView.BufferLocation = m_indexBufferResource->GetGPUVirtualAddress();
	m_indexBufferView.Format = format;
	m_indexBufferView.SizeInBytes = sizeof(UINT) * numIndices;
}

