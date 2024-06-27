struct PositionUV
{
	float3 Pos : SV_POSITION;
	float2 UV : TEXCOORD;
};

struct VertexOut
{
    float4 PosH : SV_POSITION;
    float2 UV : TEXCOORD;
};

VertexOut VS(PositionUV vertexIn)
{
    VertexOut vOut;
    vOut.PosH = float4(vertexIn.Pos, 1.0f);
    vOut.UV = vertexIn.UV;
    
    return vOut;
}