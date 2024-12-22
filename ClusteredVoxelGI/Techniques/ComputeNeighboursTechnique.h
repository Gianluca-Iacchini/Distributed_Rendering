#include "Technique.h"

namespace CVGI
{
	class ComputeNeighboursTechnique : public VOX::Technique
	{
	public:
		ComputeNeighboursTechnique(std::shared_ptr<VOX::TechniqueData> data);

		~ComputeNeighboursTechnique() {}

		void InitializeBuffers() override;
	
		void PerformTechnique(DX12Lib::ComputeContext& context) override;
		

		virtual void BuildPipelineState() override;
	protected:
		void TechniquePass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize) override;
		std::shared_ptr<DX12Lib::RootSignature> BuildRootSignature() override;

	private:
		ConstantBufferComputeNeighbour m_cbNeighbour;

	public:
		static const std::wstring Name;
		
	private:
		enum class ComputeNeighbourRootParameter
		{
			VoxelCommonsCBV = 0,
			NeighbourCBV = 1,
			ClusterVoxelsUAVTable,
			Count
		};
	};
}


