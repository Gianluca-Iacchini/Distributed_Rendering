#include "CVGIDataTypes.h"

using namespace DirectX;
using namespace CVGI;


const D3D12_INPUT_ELEMENT_DESC VertexSingleUINT::InputElements[] =
{
	{ "SV_Position", 0, DXGI_FORMAT_R32_UINT,    0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
};

static_assert(sizeof(VertexSingleUINT) == 4, "Vertex struct/layout mismatch");

const D3D12_INPUT_LAYOUT_DESC VertexSingleUINT::InputLayout =
{
	VertexSingleUINT::InputElements,
	VertexSingleUINT::InputElementCount
};