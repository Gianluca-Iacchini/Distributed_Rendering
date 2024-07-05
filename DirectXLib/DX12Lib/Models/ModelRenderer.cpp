#include "DX12Lib/pch.h"
#include "ModelRenderer.h"
#include "DX12Lib/Scene/SceneNode.h"

using namespace Graphics::Renderer;

DX12Lib::MeshRenderer::~MeshRenderer()
{
	if (m_IsInBatch)
	{
		DXLIB_CORE_WARN("MeshRenderer was removed from batch");
		if (m_modelRenderer != nullptr)
			m_modelRenderer->RemoveMeshRendererFromBatch(this);


	}
}

void DX12Lib::MeshRenderer::Init(CommandContext& context)
{
	m_objectCB = ConstantBufferObject();
	DirectX::XMStoreFloat4x4(&m_objectCB.world, DirectX::XMMatrixTranspose(Node->GetWorldMatrix()));
	DirectX::XMStoreFloat4x4(&m_objectCB.invWorld, DirectX::XMMatrixTranspose(Node->GetWorldInverse()));
	m_objectCB.texTransform = MathHelper::Identity4x4();
	m_objectCB.materialIndex = 0;
}

void DX12Lib::MeshRenderer::Update(CommandContext& context)
{
	// If is visible (not occluded, inside frustum, etc)
	m_isVisible = true;
}

void DX12Lib::MeshRenderer::Render(CommandContext& context)
{

	if (m_modelRenderer == nullptr)
		return;

	if (m_isVisible)
	{
		if (!m_IsInBatch)
		{
			m_modelRenderer->AddMeshRendererToBatch(this);
			m_IsInBatch = true;
		}
	}

	else if (!m_isVisible)
	{
		if (m_IsInBatch)
		{
			m_modelRenderer->RemoveMeshRendererFromBatch(this);
			m_IsInBatch = false;
		}
	}

}

void DX12Lib::MeshRenderer::DrawMesh(CommandContext& context)
{

	if (m_mesh == nullptr)
		return;

	m_mesh->Draw(context.m_commandList->Get());
}

void DX12Lib::MeshRenderer::SetMaterial(Material* material)
{
	assert(material != nullptr);

	if (material->GetName() != m_meshMaterial->GetName())
	{
		if (m_modelRenderer != nullptr)
		{
			m_meshMaterial = material;
			m_modelRenderer->UpdateMeshRendererBatch(this);
			m_IsInBatch = true;
		}
	}
}

DirectX::GraphicsResource DX12Lib::MeshRenderer::GetObjectCB()
{

	DirectX::XMStoreFloat4x4(&m_objectCB.world, DirectX::XMMatrixTranspose(Node->GetWorldMatrix()));
	DirectX::XMStoreFloat4x4(&m_objectCB.invWorld, DirectX::XMMatrixTranspose(Node->GetWorldInverse()));
	m_objectCB.texTransform = MathHelper::Identity4x4();
	m_objectCB.materialIndex = m_mesh->MeshMaterial->GetIndex();
	return Graphics::Renderer::s_graphicsMemory->AllocateConstant(m_objectCB);
}

const DX12Lib::DescriptorHandle& DX12Lib::MeshRenderer::GetMaterialTextureSRV()
{
	assert(m_meshMaterial != nullptr && m_meshMaterial->GetTextureCount() > 0);

	return m_meshMaterial->GetFirstTextureHandle();
}

void DX12Lib::ModelRenderer::Render(CommandContext& context)
{
	if (Model != nullptr)
	{
		Graphics::Renderer::AddRendererToQueue(this);
	}
}


std::vector<DX12Lib::MeshRenderer*> DX12Lib::ModelRenderer::GetAllForPSO(std::wstring psoName)
{
	if (Model == nullptr)
		return std::vector<DX12Lib::MeshRenderer*>();

	auto batch = m_psoMeshRendererBatch.find(psoName);

	if (batch == m_psoMeshRendererBatch.end())
		return std::vector<DX12Lib::MeshRenderer*>();

	return batch->second;
}

std::vector<DX12Lib::MeshRenderer*> DX12Lib::ModelRenderer::GetAll()
{
	std::vector<DX12Lib::MeshRenderer*> allMeshes;

	for (auto& batch : m_psoMeshRendererBatch)
	{
		allMeshes.insert(allMeshes.end(), batch.second.begin(), batch.second.end());
	}

	return allMeshes;
}

std::vector<DX12Lib::MeshRenderer*> DX12Lib::ModelRenderer::GetAllOpaqueForPSO(std::wstring psoName)
{

	if (Model == nullptr)
		return std::vector<DX12Lib::MeshRenderer*>();

	auto batch = m_psoMeshRendererBatch.find(psoName);

	if (batch == m_psoMeshRendererBatch.end())
		return std::vector<DX12Lib::MeshRenderer*>();

	std::vector<MeshRenderer*> opaqueMeshes;

	for (auto& mesh : batch->second)
	{
		if (!mesh->GetMaterial()->IsTransparent())
			opaqueMeshes.push_back(mesh);
	}

	return opaqueMeshes;
}

std::vector<DX12Lib::MeshRenderer*> DX12Lib::ModelRenderer::GetAllTransparentForPSO(std::wstring psoName)
{
	if (Model == nullptr)
		return std::vector<DX12Lib::MeshRenderer*>();

	auto batch = m_psoMeshRendererBatch.find(psoName);

	if (batch == m_psoMeshRendererBatch.end())
		return std::vector<DX12Lib::MeshRenderer*>();

	std::vector<MeshRenderer*> transparentMeshes;

	for (auto& mesh : batch->second)
	{
		if (mesh->GetMaterial()->IsTransparent())
			transparentMeshes.push_back(mesh);
	}

	return transparentMeshes;
}


std::vector<DX12Lib::MeshRenderer*> DX12Lib::ModelRenderer::GetAllOpaque()
{
	return m_opaqueMeshes;
}

std::vector<DX12Lib::MeshRenderer*> DX12Lib::ModelRenderer::GetAllTransparent()
{
	return m_transparentMeshes;
}
