float4 VS( ) : SV_POSITION
{
    return float4(1.0f, 1.0f, 1.0f, 1.0f);
}

float4 PS(float4 pIn : SV_Position) : SV_TARGET
{
    return float4(1.0f, 0.0f, 0.0f, 1.0f);
}