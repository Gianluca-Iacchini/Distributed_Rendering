#include "Helpers.h"

#ifndef PIPELINE_STATE_H
#define PIPELINE_STATE_H

class PipelineState
{
public:
	PipelineState() {};
	~PipelineState() {};

	Microsoft::WRL::ComPtr<ID3D12PipelineState> Get() const { return m_pipelineState; }
	ID3D12PipelineState** GetAddressOf() { return m_pipelineState.GetAddressOf(); }
	Microsoft::WRL::ComPtr<ID3D12PipelineState> GetComPtr() const { return m_pipelineState; }
private:
	Microsoft::WRL::ComPtr<ID3D12PipelineState> m_pipelineState;
};

#endif // !PIPELINE_STATE_H



