

float4 VS(float3 Pos : SV_Position) : SV_Position
{
    return float4(Pos, 1.0f);
}