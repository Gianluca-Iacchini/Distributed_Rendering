#include "DX12Lib/pch.h"

#include "Model.h"
#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "Mesh.h"



using namespace DX12Lib;

using namespace Assimp;
using namespace Graphics;

void Model::LoadFromFile(const aiScene* scene)
{
    this->LoadMaterials(scene);
    this->LoadMeshes(scene);
}

void Model::LoadMaterials(const aiScene* scene)
{
    m_materials.resize(scene->mNumMaterials);

    int phongIndex = 0;
    int pbrIndex = 0;

    for (UINT i = 0; i < scene->mNumMaterials; i++)
    {
        aiMaterial* material = scene->mMaterials[i];

        auto builder = s_materialManager->CreateMaterialBuilder();
        m_materials[i] = builder.BuildFromAssimpMaterial(material);

        // Mapping for materials structured buffer indices

        int index = builder.IsPBR() ? pbrIndex++ : phongIndex++;

        m_materialIndexMap[i] = index;
    }

    m_phongMaterialCount = phongIndex;
    m_PBRMaterialCount = pbrIndex;

    BuildMaterialStructuredBuffers();
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
        // newMesh->m_materialIndex = m_materialIndexMap[assimpMesh->mMaterialIndex];

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


    if (m_PBRMaterialCount > 0)
        commandList->SetGraphicsRootShaderResourceView(3, m_pbrMaterialBuffer->GetGPUVirtualAddress());

 //   for (auto mesh : m_meshes)
 //   {
 //       SharedMaterial mat = m_materials[mesh->m_materialIndex];
 //       mat->UseMaterial(commandList);
 //       mesh->Draw(commandList);
	//}
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

void DX12Lib::Model::BuildMaterialStructuredBuffers()
{
    UINT phongMaterialSize = sizeof(ConstantBufferPhongMaterial) * m_phongMaterialCount;
    UINT pbrMaterialSize = sizeof(ConstantBufferPBRMaterial) * m_PBRMaterialCount;

    std::vector<ConstantBufferPhongMaterial> phongMaterials(m_phongMaterialCount);
    std::vector<ConstantBufferPBRMaterial> pbrMaterials(m_PBRMaterialCount);

    for (SharedMaterial mat : m_materials)
    {
        PhongMaterial* phongMat = dynamic_cast<PhongMaterial*>(mat.get());

        if (phongMat != nullptr)
        {
            auto cb = phongMat->CreatePhongMaterialBuffer();
            phongMaterials.push_back(cb);
        }
        else
        {
            PBRMaterial* pbrMat = dynamic_cast<PBRMaterial*>(mat.get());

            if (pbrMat != nullptr)
            {
				auto cb = pbrMat->CreatePBRMaterialBuffer();
				pbrMaterials.push_back(cb);
			}
            else
            {
                DXLIB_CORE_ERROR(L"Material is not of type Phong or PBR");
            }
        }
    }

    if (m_phongMaterialCount > 0)
        m_phongMaterialBuffer = Utils::CreateDefaultBuffer(phongMaterials.data(), phongMaterialSize);
    if (m_PBRMaterialCount > 0)
        m_pbrMaterialBuffer = Utils::CreateDefaultBuffer(pbrMaterials.data(), pbrMaterialSize);
}
