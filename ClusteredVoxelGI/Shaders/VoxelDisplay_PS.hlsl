struct PSInput
{
	float4 position : SV_POSITION;
    float3 normal : NORMAL;
	float3 color : COLOR;
	uint ClusterIndex : CLUSTERINDEX;
};

float4 PS(PSInput psIn) : SV_TARGET
{
	return float4(psIn.color, psIn.ClusterIndex);
}