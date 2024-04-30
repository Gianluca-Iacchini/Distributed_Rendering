#include "Model.h"
#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"
#include "iostream"
#include "Mesh.h"
#include "GraphicsCore.h"
#include "CommandList.h"
#include "CommandContext.h"

using namespace Assimp;
using namespace Graphics;

bool Model::LoadFromFile(const std::wstring filename)
{
    Importer importer;

    std::string filenameStr = Utils::ToString(filename);

    importer.SetPropertyFloat(AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE, 80.0f);
    importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_POINT | aiPrimitiveType_LINE);

    unsigned int preprocessFlags = aiProcessPreset_TargetRealtime_MaxQuality | aiProcess_OptimizeGraph |
        aiProcess_ConvertToLeftHanded | aiProcess_GenBoundingBoxes;

    const aiScene* scene = importer.ReadFile(filenameStr, preprocessFlags);

    assert(scene != nullptr && "Failed to load scene");

    UINT numMeshes = scene->mNumMeshes;

    for (UINT i = 0; i < numMeshes; i++)
    {
        auto mesh = scene->mMeshes[i];

        std::shared_ptr<Mesh> newMesh = std::make_shared<Mesh>();
        
        std::vector<DirectX::VertexPositionNormalTexture> vertices(mesh->mNumVertices);


        for (UINT v = 0; v < mesh->mNumVertices; v++)
        {
            if (mesh->HasPositions())
                vertices[v].position = { mesh->mVertices[v].x, mesh->mVertices[v].y, mesh->mVertices[v].z };
            
            if (mesh->HasNormals())
                vertices[v].normal = { mesh->mNormals[v].x, mesh->mNormals[v].y, mesh->mNormals[v].z };
            
            if (mesh->HasTextureCoords(0))
                vertices[v].textureCoordinate = { mesh->mTextureCoords[0][v].x, mesh->mTextureCoords[0][v].y };
        }

        newMesh->m_vertexBufferByteSize = sizeof(DirectX::VertexPositionNormalTexture) * mesh->mNumVertices;
        newMesh->m_vertexBufferStride = sizeof(DirectX::VertexPositionNormalTexture);
        newMesh->m_vertexBufferResource = Utils::CreateDefaultBuffer(s_device->GetComPtr(), vertices.data(), newMesh->m_vertexBufferByteSize);

        assert(mesh->HasFaces() && "Mesh has no faces");

        std::vector<UINT> indices;
        UINT numFaces = mesh->mNumFaces;

        for (UINT j = 0; j < numFaces; j++)
        {
            const aiFace& face = mesh->mFaces[j];

            if (face.mNumIndices == 3)
            {
                indices.push_back(face.mIndices[0]);
                indices.push_back(face.mIndices[1]);
                indices.push_back(face.mIndices[2]);
			}
            //assert(face.mNumIndices == 3 && "Mesh face doesn't have 3 indices");

        }
        
        DXGI_FORMAT indexFormat = DXGI_FORMAT_R32_UINT;

        newMesh->m_indexBufferFormat = indexFormat;
        newMesh->m_numIndices = indices.size();
        newMesh->m_indexBufferByteSize = sizeof(UINT) * newMesh->m_numIndices;
        newMesh->m_indexBufferResource = Utils::CreateDefaultBuffer(s_device->GetComPtr(), indices.data(), newMesh->m_indexBufferByteSize);

        m_meshes.push_back(newMesh);
    }

    return true;
}

bool Model::LoadFromFile(const char* filename)
{
    return LoadFromFile(Utils::ToWstring(filename));
}

void Model::Draw(ID3D12GraphicsCommandList* commandList)
{
    assert(commandList != nullptr && "CommandList is null");

    for (auto mesh : m_meshes)
    {
        auto vertexBufferView = mesh->VertexBufferView();
		commandList->IASetVertexBuffers(0, 1, &vertexBufferView);
        auto indexBufferView = mesh->IndexBufferView();
		commandList->IASetIndexBuffer(&indexBufferView);
		commandList->IASetPrimitiveTopology(mesh->m_primitiveTopology);
		commandList->DrawIndexedInstanced(mesh->m_numIndices, 1, 0, 0, 0);
	}
}

void Model::Draw(CommandList& commandList)
{
    Draw(commandList.Get());
}

void Model::Draw(CommandContext& context)
{
    assert(context.m_commandList != nullptr && "CommandList is null");
	Draw(*context.m_commandList);
}
