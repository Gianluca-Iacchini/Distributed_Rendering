#include "DX12Lib/pch.h"
#include "ModelRenderer.h"
#include "DX12Lib/Scene/SceneNode.h"

using namespace Graphics::Renderer;

DX12Lib::MeshRenderer::~MeshRenderer()
{
	if (m_IsInBatch)
	{
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

	if (m_mesh == nullptr || m_meshMaterial == nullptr)
		return;

}

void DX12Lib::MeshRenderer::DrawMesh(CommandContext& context)
{

	if (m_mesh == nullptr)
		return;

	if (m_meshMaterial != nullptr)
		m_meshMaterial->UseMaterial(context.m_commandList->Get());

	context.m_commandList->Get()->SetGraphicsRootConstantBufferView((UINT)RootSignatureSlot::ObjectCBV, GetObjectCB().GpuAddress());

	m_mesh->Draw(context.m_commandList->Get());
}

DirectX::GraphicsResource DX12Lib::MeshRenderer::GetObjectCB()
{

	DirectX::XMStoreFloat4x4(&m_objectCB.world, DirectX::XMMatrixTranspose(Node->GetWorldMatrix()));
	DirectX::XMStoreFloat4x4(&m_objectCB.invWorld, DirectX::XMMatrixTranspose(Node->GetWorldInverse()));
	m_objectCB.texTransform = MathHelper::Identity4x4();
	m_objectCB.materialIndex = m_mesh->m_materialIndex;
	return Graphics::Renderer::s_graphicsMemory->AllocateConstant(m_objectCB);
}

void DX12Lib::ModelRenderer::Render(CommandContext& context)
{
	if (Model != nullptr)
	{
		Graphics::Renderer::AddRendererToQueue(this);
	}
}

void DX12Lib::ModelRenderer::DrawMeshes(CommandContext& context, std::vector<MeshRenderer*> meshRenderers)
{
	Model->UseBuffers(context);
	for (auto& mesh : meshRenderers)
	{
		mesh->DrawMesh(context);
	}
}

void DX12Lib::ModelRenderer::DrawAllBatch(CommandContext& context, std::wstring psoName)
{
	if (Model == nullptr)
		return;

	auto batch = m_psoMeshRendererBatch.find(psoName);

	if (batch == m_psoMeshRendererBatch.end())
		return;

	DrawMeshes(context, batch->second);
}

void DX12Lib::ModelRenderer::DrawAll(CommandContext& context)
{
	for (auto& batch : m_psoMeshRendererBatch)
	{
		DrawAllBatch(context, batch.first);
	}
}

void DX12Lib::ModelRenderer::DrawBatchOpaque(CommandContext& context, std::wstring psoName)
{

	if (Model == nullptr)
		return;

	auto batch = m_psoMeshRendererBatch.find(psoName);

	if (batch == m_psoMeshRendererBatch.end())
		return;

	std::vector<MeshRenderer*> opaqueMeshes;

	for (auto& mesh : batch->second)
	{
		if (!mesh->GetMaterial()->IsTransparent())
			opaqueMeshes.push_back(mesh);
	}

	DrawMeshes(context, opaqueMeshes);
}

void DX12Lib::ModelRenderer::DrawBatchTransparent(CommandContext& context, std::wstring psoName)
{
	if (Model == nullptr)
		return;

	auto batch = m_psoMeshRendererBatch.find(psoName);

	if (batch == m_psoMeshRendererBatch.end())
		return;

	std::vector<MeshRenderer*> transparentMeshes;

	for (auto& mesh : batch->second)
	{
		if (mesh->GetMaterial()->IsTransparent())
			transparentMeshes.push_back(mesh);
	}

	DrawMeshes(context, transparentMeshes);
}


void DX12Lib::ModelRenderer::DrawOpaque(CommandContext& context)
{
	DrawMeshes(context, m_opaqueMeshes);
}

void DX12Lib::ModelRenderer::DrawTransparent(CommandContext& context)
{
	DrawMeshes(context, m_transparentMeshes);
}
