#include "Model.h"
#include "assimp/Importer.hpp"
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

    unsigned int preprocessFlags = aiProcess_ConvertToLeftHanded | aiProcessPreset_TargetRealtime_MaxQuality | aiProcess_Triangulate; 
    /*| aiProcess_OptimizeGraph | aiProcess_OptimizeMeshes | aiProcess_GenBoundingBoxes;*/

    const aiScene* scene = importer.ReadFile(filenameStr, preprocessFlags);

    assert(scene != nullptr && "Failed to load scene");

    this->LoadTexture(scene);

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

        newMesh->m_materialIndex = assimpMesh->mMaterialIndex;
        
        if (newMesh->m_materialIndex == 12)
            newMesh->m_materialIndex = 0;
        
        m_meshes.push_back(newMesh);
    }

    UINT vertexStride = sizeof(DirectX::VertexPositionNormalTexture);
    m_vertexBufferResource = Utils::CreateDefaultBuffer(s_device->GetComPtr(), vertices.data(), vertexStride * vertices.size());
    BuildVertexBuffer(vertexStride, vertices.size());
    
    m_indexBufferResource = Utils::CreateDefaultBuffer(s_device->GetComPtr(), indices.data(), sizeof(UINT) * indices.size());
    BuildIndexBuffer(DXGI_FORMAT_R32_UINT, indices.size());

    return true;
}

bool Model::LoadFromFile(const char* filename)
{
    return LoadFromFile(Utils::ToWstring(filename));
}

void Model::LoadTexture(const aiScene* scene)
{
    m_textureHeap.Create(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4096);

    m_materials = std::vector<std::pair<SharedTexture, DescriptorHandle>>(scene->mNumMaterials);

    for (UINT i = 0; i < scene->mNumMaterials; i++)
    {
        aiMaterial* material = scene->mMaterials[i];

        UINT16 diffuseCount = material->GetTextureCount(aiTextureType_DIFFUSE);

        if (diffuseCount > 0)
        {
            aiString texturePath;
            material->GetTexture(aiTextureType_DIFFUSE, 0, &texturePath);

            std::wstring texturePathW = Utils::ToWstring(texturePath.C_Str());

            SharedTexture texture = s_textureManager->LoadFromFile(ModelFolder + L"\\" + texturePathW, false);
            DescriptorHandle handle = m_textureHeap.Alloc(1);

            s_device->GetComPtr()->CopyDescriptorsSimple(1, handle, texture->GetSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

            m_materials[i] = std::make_pair(texture, handle);
        }
        else
        {
            m_materials[i] = std::make_pair(nullptr, DescriptorHandle());
        }
    }
}

void Model::Draw(ID3D12GraphicsCommandList* commandList)
{
    assert(commandList != nullptr && "CommandList is null");

    commandList->IASetVertexBuffers(0, 1, &m_vertexBufferView);
    commandList->IASetIndexBuffer(&m_indexBufferView);

    ID3D12DescriptorHeap* heaps[] = { m_textureHeap.Get() };
    commandList->SetDescriptorHeaps(1, heaps);

    for (auto mesh : m_meshes)
    {
        DescriptorHandle handle = m_materials[mesh->m_materialIndex].second;
        commandList->SetGraphicsRootDescriptorTable(2, handle);
		commandList->IASetPrimitiveTopology(mesh->m_primitiveTopology);
		commandList->DrawIndexedInstanced(mesh->m_numIndices, 1, mesh->m_indexStart, mesh->m_vertexStart, 0);
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
