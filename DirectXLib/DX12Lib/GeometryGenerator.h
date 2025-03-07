#pragma once
#include <cstdint>
#include <DirectXMath.h>

namespace DX12Lib {

	class GeometryGenerator
	{
	public:

		struct Vertex
		{
			Vertex() {}

			Vertex(
				const DirectX::XMFLOAT3& p,
				const DirectX::XMFLOAT3& n,
				const DirectX::XMFLOAT3& t,
				const DirectX::XMFLOAT2& uv
			)
				: Position(p),
				Normal(n),
				TangentU(t),
				TexC(uv) {}

			Vertex(
				float px, float py, float pz,
				float nx, float ny, float nz,
				float tx, float ty, float tz,
				float u, float v
			)
				: Position(px, py, pz),
				Normal(nx, ny, nz),
				TangentU(tx, ty, tz),
				TexC(u, v) {}

			DirectX::XMFLOAT3 Position;
			DirectX::XMFLOAT3 Normal;
			DirectX::XMFLOAT3 TangentU;
			DirectX::XMFLOAT2 TexC;
		};

		struct MeshData
		{
		public:
			std::vector<Vertex> Vertices;
			std::vector<uint32_t> Indices32;

			std::vector<uint16_t>& GetIndices16()
			{
				if (m_Indices16.empty())
				{
					m_Indices16.resize(Indices32.size());
					for (size_t i = 0; i < Indices32.size(); ++i)
					{
						m_Indices16[i] = static_cast<uint16_t>(Indices32[i]);
					}
				}

				return m_Indices16;
			}

		private:
			std::vector<uint16_t> m_Indices16;
		};

		MeshData CreateBox(float width, float height, float depth, uint32_t numSubdivisions);

		MeshData CreateSphere(float radius, uint32_t sliceCount, uint32_t stackCount);

		MeshData CreateGeosphere(float radius, uint32_t numSubdivisions);

		MeshData CreateCylinder(float bottomRadius, float topRadius, float height, uint32_t sliceCount, uint32_t stackCount);

		MeshData CreateGrid(float width, float depth, uint32_t m, uint32_t n);

		MeshData CreateQuad(float x, float y, float w, float h, float depth);

	private:

		void Subdivide(MeshData& meshData);
		Vertex MidPoint(const Vertex& v0, const Vertex& v1);
		void BuildCylinderTopCap(float bottomRadius, float topRadius, float height, uint32_t sliceCount, uint32_t stackCount, MeshData& meshData);
		void BuildCylinderBottomCap(float bottomRadius, float topRadius, float height, uint32_t sliceCount, uint32_t stackCount, MeshData& meshData);
	};
}
