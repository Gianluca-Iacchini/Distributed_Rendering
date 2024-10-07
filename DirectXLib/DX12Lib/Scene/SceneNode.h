#pragma once
#include "Transform.h"
#include "Component.h"
#include "DX12Lib/Models/ModelRenderer.h"


namespace DX12Lib
{
	class SceneNode;
	class Scene;

	using NodePtr = std::unique_ptr<SceneNode>;

	class SceneNode
	{
	public:
		SceneNode(DX12Lib::Scene& scene) : m_dirtyFlags(0), Scene(scene)
		{
			ThrowIfFailed(CoCreateGuid(&m_guid));
			m_name = L"Node_" + std::to_wstring(s_numNodes++);
			s_numNodes++;
		}
		~SceneNode();

		void Init(CommandContext&);
		void Update(CommandContext& context);
		void Render(CommandContext& context);
		void OnResize(CommandContext&, int newWidth, int newHeight);
		void OnClose(CommandContext& context);

		Scene& GetScene() { return Scene; }
		void AddChild(SceneNode* node);
		SceneNode* AddChild();
		void RemoveChild(SceneNode* node);
		SceneNode* GetChild(SceneNode* node);
		std::vector<Component*> GetComponents();
		
		template<typename T>
		T* GetComponent()
		{
			static_assert(std::is_base_of<Component, T>::value, "T must derive from Component");

			for (auto& component : m_components)
			{
				T* castedComponent = dynamic_cast<T*>(component.get());
				if (castedComponent != nullptr)
				{
					return castedComponent;
				}
			}

			return nullptr;
		}

		template<typename T, typename... Args>
		inline T* AddComponent(Args&&... args)
		{
			static_assert(std::is_base_of<Component, T>::value, "T must derive from Component");

			std::unique_ptr<T> componentPtr = std::make_unique<T>(std::forward<Args>(args)...);
			componentPtr->Node = this;
			T* returnPtr = componentPtr.get();

			this->m_components.push_back(std::move(componentPtr));

			return returnPtr;
		}

		void RemoveComponent(Component* component);

		void SetPosition(const DirectX::XMFLOAT3& position);
		void SetPosition(float x, float y, float z);
		void SetRotationQuaternion(const DirectX::XMFLOAT4& rotationQuat);
		void SetRotationEulerAngles(const DirectX::XMFLOAT3& rotationEuler);
		void SetRotationEulerAngles(float pitch, float yaw, float roll);
		void SetScale(const DirectX::XMFLOAT3& scale);
		void SetScale(float x, float y, float z);

		void SetRelativePosition(const DirectX::XMFLOAT3& position);
		void SetRelativePosition(float x, float y, float z);
		void SetRelativeRotation(const DirectX::XMFLOAT4& rotationQuat);
		void SetRelativeRotation(const DirectX::XMFLOAT3& rotationEuler);
		void SetRelativeRotation(float pitch, float yaw, float roll);
		void SetRelativeScale(const DirectX::XMFLOAT3& scale);
		void SetRelativeScale(float x, float y, float z);

		void Translate(const DirectX::XMFLOAT3& translation, float value = 1.0f);
		void Translate(float x, float y, float z);
		void Rotate(const DirectX::XMFLOAT3& axis, float value = 1.0f);
		void Rotate(float pitch, float yaw, float roll);
		void LookAt(const DirectX::XMFLOAT3& target, const DirectX::XMFLOAT3& up = DirectX::XMFLOAT3(0, 1, 0));

		inline DirectX::XMFLOAT3 GetForward(){ return this->Transform.GetForward3f();}
		inline DirectX::XMFLOAT3 GetRight()  { return this->Transform.GetRight3f(); }
		inline DirectX::XMFLOAT3 GetUp() { return this->Transform.GetUp3f(); }

		DirectX::XMFLOAT3 GetPosition();
		DirectX::XMFLOAT3 GetRelativePosition();
		DirectX::XMFLOAT4 GetRotationQuaternion();
		DirectX::XMFLOAT4 GetRelativeRotationQuaternion();
		DirectX::XMFLOAT3 GetRotationEulerAngles();
		DirectX::XMFLOAT3 GetRelativeRotationEulerAngles();
		DirectX::XMFLOAT3 GetScale();
		DirectX::XMFLOAT3 GetRelativeScale();

		DirectX::XMFLOAT4X4 GetWorldMatrix4x4();
		DirectX::XMMATRIX GetWorldMatrix();
		DirectX::XMFLOAT4X4 GetWorldInverse4x4();
		DirectX::XMMATRIX GetWorldInverse();

		int GetChildCount() const { return m_children.size(); }

		SceneNode* GetChildAt(UINT index) const 
		{ 
			assert(index < m_children.size());

			return m_children[index].get(); 
		}

		bool IsTransformDirty() const 
		{ 
			return (this->Transform.m_dirtForFrame > 0) || m_wasLastFrameDirty; 
		}

		void SetName(const std::wstring& name) { m_name = name; }
		const std::wstring& GetName() const { return m_name; }

	public:
		Transform Transform;
		Scene& Scene;
	
	public:
		bool operator==(const SceneNode& other) const { return m_guid == other.m_guid; }

	private:
		void PropagateDirtyTransform(UINT dirtyFlag = 5);

	private:


		void SetParent(SceneNode* parent);

		// Raw pointer because child nodes do not own their parent
		SceneNode* m_parent = nullptr;
		std::vector<NodePtr> m_children;

		UINT m_dirtyFlags;

		enum class ParentDirtyFlags
		{
			Position = 1 << 0,
			Rotation = 1 << 1,
			Scale = 1 << 2
		};


		std::vector<std::unique_ptr<Component>> m_components;


	private:
		GUID m_guid = GUID_NULL;
		std::wstring m_name = L"";
		bool m_wasLastFrameDirty = false;

		static UINT64 s_numNodes;
	};




}

