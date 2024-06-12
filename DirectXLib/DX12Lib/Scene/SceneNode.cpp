#include "DX12Lib/pch.h"
#include "SceneNode.h"
#include "Scene.h"

using namespace DX12Lib;

DX12Lib::SceneNode::~SceneNode()
{
	// Manually release the vector so the children are deleted before the parent
	// This is important because the the children may refer the parent components
	m_children.clear();
	m_components.clear();
}

void DX12Lib::SceneNode::Init(CommandContext& context)
{
	for (std::unique_ptr<Component>& component : m_components)
	{
		component->Init(context);
	}

	for (auto& node : m_children)
		node->Init(context);
}

void DX12Lib::SceneNode::Update(CommandContext& context)
{
	// Will only update if the transform is dirty
	Transform.Update();


	for (std::unique_ptr<Component>& component : m_components)
	{
		component->Update(context);
	}
	

	for (auto& node : m_children) 
		node->Update(context);

	// Reset the dirt flag for the frame
	this->Transform.m_dirtForFrame = 0;
}

void DX12Lib::SceneNode::Render(CommandContext& context)
{


	for (std::unique_ptr<Component>& component : m_components)
	{
		component->Render(context);
	}


	for (auto& node : m_children)
		node->Render(context);
	
}

void DX12Lib::SceneNode::OnResize(CommandContext& context, int newWidth, int newHeight)
{
	for (std::unique_ptr<Component>& component : m_components)
	{
		component->OnResize(context, newWidth, newHeight);
	}

	for (auto& node : m_children)
		node->OnResize(context, newWidth, newHeight);
}

void DX12Lib::SceneNode::AddChild(SceneNode* node)
{
	assert(node != nullptr);

	if (node->m_parent != nullptr)
	{
		auto& children = node->m_parent->m_children;

		auto iter = std::find_if(children.begin(), children.end(),
			[&node](const std::unique_ptr<SceneNode>& child)
			{
				return *child == *node;
			});

		if (iter != children.end())
		{
			this->m_children.push_back(std::move(*iter));
			children.erase(iter);
		}
	}

	node->SetParent(this);
}

SceneNode* DX12Lib::SceneNode::AddChild()
{
	std::unique_ptr<SceneNode> node = std::make_unique<SceneNode>(Scene);
	SceneNode* nodePtr = node.get();

	m_children.push_back(std::move(node));

	nodePtr->SetParent(this);

	return nodePtr;
}



void DX12Lib::SceneNode::RemoveChild(SceneNode* node)
{
	auto iter = std::find_if(m_children.begin(), m_children.end(),
		[node](const std::unique_ptr<DX12Lib::SceneNode>& child) {
			return child.get() == node;
		});

	if (iter != m_children.end())
	{
		m_children.erase(iter);
	}
}

SceneNode* DX12Lib::SceneNode::GetChild(SceneNode* node)
{
	auto iter = std::find_if(m_children.begin(), m_children.end(),
		[node](const std::unique_ptr<DX12Lib::SceneNode>& child) {
			return child.get() == node;
		});

	if (iter != m_children.end())
	{
		return iter->get();
	}

	return nullptr;
}

void DX12Lib::SceneNode::PropagateDirtyTransform(UINT dirtyFlag)
{
	Transform.SetDirty(dirtyFlag);

	for (auto& node : m_children)
	{
		node->PropagateDirtyTransform(dirtyFlag);
	}
}

void SceneNode::SetParent(SceneNode* parent)
{
	this->m_parent = parent;

	if (parent != nullptr)
	{
		Transform.m_parent = &parent->Transform;
	}

	PropagateDirtyTransform();
}


void DX12Lib::SceneNode::SetPosition(const DirectX::XMFLOAT3& position)
{
	Transform.SetWorldPosition(position);
	PropagateDirtyTransform((UINT)Transform::DirtyFlags::Position);
}

void DX12Lib::SceneNode::SetPosition(float x, float y, float z)
{
	this->SetPosition(DirectX::XMFLOAT3(x, y, z));
}

void DX12Lib::SceneNode::SetRotationQuaternion(const DirectX::XMFLOAT4& quatRotation)
{
	Transform.SetWorldRotation(quatRotation);
	PropagateDirtyTransform((UINT)Transform::DirtyFlags::Rotation);
}

void DX12Lib::SceneNode::SetRotationEulerAngles(const DirectX::XMFLOAT3& rotationEuler)
{
	DirectX::XMVECTOR rotation = DirectX::XMQuaternionRotationRollPitchYaw(rotationEuler.x, rotationEuler.y, rotationEuler.z);
	Transform.SetWorldRotation(rotation);
	PropagateDirtyTransform((UINT)Transform::DirtyFlags::Rotation);
}

void DX12Lib::SceneNode::SetRotationEulerAngles(float pitch, float yaw, float roll)
{
	this->SetRotationEulerAngles(DirectX::XMFLOAT3(pitch, yaw, roll));
}

void DX12Lib::SceneNode::SetScale(const DirectX::XMFLOAT3& scale)
{
	Transform.SetWorldScale(scale);
	PropagateDirtyTransform((UINT)Transform::DirtyFlags::Scale);
}

void DX12Lib::SceneNode::SetScale(float x, float y, float z)
{
	this->SetScale(DirectX::XMFLOAT3(x, y, z));
}

void DX12Lib::SceneNode::SetRelativePosition(const DirectX::XMFLOAT3& position)
{
	Transform.SetRelativePosition(position);
	PropagateDirtyTransform((UINT)Transform::DirtyFlags::Position);
}

void DX12Lib::SceneNode::SetRelativePosition(float x, float y, float z)
{
	this->SetRelativePosition(DirectX::XMFLOAT3(x, y, z));
}

void DX12Lib::SceneNode::SetRelativeRotation(const DirectX::XMFLOAT4& quatRotation)
{
	Transform.SetRelativeRotation(quatRotation);
	PropagateDirtyTransform((UINT)Transform::DirtyFlags::Rotation);
}

void DX12Lib::SceneNode::SetRelativeRotation(const DirectX::XMFLOAT3& rotationEuler)
{
	DirectX::XMVECTOR rotation = DirectX::XMQuaternionRotationRollPitchYaw(rotationEuler.x, rotationEuler.y, rotationEuler.z);
	Transform.SetRelativeRotation(rotation);
	PropagateDirtyTransform((UINT)Transform::DirtyFlags::Rotation);
}

void DX12Lib::SceneNode::SetRelativeRotation(float pitch, float yaw, float roll)
{
	this->SetRelativeRotation(DirectX::XMFLOAT3(pitch, yaw, roll));
}

void DX12Lib::SceneNode::SetRelativeScale(const DirectX::XMFLOAT3& scale)
{
	Transform.SetRelativeScale(scale);
	PropagateDirtyTransform((UINT)Transform::DirtyFlags::Scale);
}

void DX12Lib::SceneNode::SetRelativeScale(float x, float y, float z)
{
	this->SetRelativeScale(DirectX::XMFLOAT3(x, y, z));
}

DirectX::XMFLOAT3 DX12Lib::SceneNode::GetPosition()
{
	return Transform.GetWorldPosition3f();
}

DirectX::XMFLOAT3 DX12Lib::SceneNode::GetRelativePosition()
{
	return Transform.GetRelativePosition3f();
}

DirectX::XMFLOAT4 DX12Lib::SceneNode::GetRotationQuaternion()
{
	return Transform.GetWorldRotation4f();
}

DirectX::XMFLOAT4 DX12Lib::SceneNode::GetRelativeRotationQuaternion()
{
	return Transform.GetRelativeRotation4f();
}

DirectX::XMFLOAT3 DX12Lib::SceneNode::GetRotationEulerAngles()
{
	return Transform.GetWorldRotationEuler3f();
}

DirectX::XMFLOAT3 DX12Lib::SceneNode::GetRelativeRotationEulerAngles()
{
	return Transform.GetRelativeRotationEuler3f();
}

DirectX::XMFLOAT3 DX12Lib::SceneNode::GetScale()
{
	return Transform.GetWorldScale3f();
}

DirectX::XMFLOAT3 DX12Lib::SceneNode::GetRelativeScale()
{
	return Transform.GetRelativeScale3f();
}

void DX12Lib::SceneNode::Translate(float x, float y, float z)
{

	DirectX::XMVECTOR position = this->Transform.GetRelativePosition();
	position = DirectX::XMVectorAdd(position, DirectX::XMVectorSet(x, y, z, 0.0f));
	this->Transform.SetRelativePosition(position);
}

void DX12Lib::SceneNode::Translate(const DirectX::XMFLOAT3& translation, float value)
{
	this->Translate(translation.x * value, translation.y * value, translation.z * value);
}

void DX12Lib::SceneNode::Rotate(float pitch, float yaw, float roll)
{
	DirectX::XMVECTOR axis = DirectX::XMVectorSet(pitch, yaw, roll, 0.0f);
	DirectX::XMVECTOR quaternion = this->Transform.GetRelativeRotation();

	quaternion = DirectX::XMQuaternionMultiply(quaternion, DirectX::XMQuaternionRotationRollPitchYaw(pitch, yaw, roll));
	
	quaternion = DirectX::XMQuaternionNormalize(quaternion);
	

	this->Transform.SetRelativeRotation(quaternion);
}

void DX12Lib::SceneNode::Rotate(const DirectX::XMFLOAT3& axis, float value)
{
	this->Rotate(axis.x * value, axis.y * value , axis.z * value);
}


DirectX::XMFLOAT4X4 DX12Lib::SceneNode::GetWorldMatrix4x4()
{
	DirectX::XMFLOAT4X4 world;
	DirectX::XMStoreFloat4x4(&world, Transform.GetWorld());

	return world;
}


DirectX::XMFLOAT4X4 DX12Lib::SceneNode::GetWorldInverse4x4()
{
	DirectX::XMFLOAT4X4 world4x4;
	DirectX::XMMATRIX world = DirectX::XMMatrixInverse(nullptr, Transform.GetWorld());
	DirectX::XMStoreFloat4x4(&world4x4, world);

	return world4x4;
}

DirectX::XMMATRIX DX12Lib::SceneNode::GetWorldMatrix()
{
	return Transform.GetWorld();
}

DirectX::XMMATRIX DX12Lib::SceneNode::GetWorldInverse()
{
	return DirectX::XMMatrixInverse(nullptr, Transform.GetWorld());
}

