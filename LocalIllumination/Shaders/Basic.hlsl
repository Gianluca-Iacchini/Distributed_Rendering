float gTime : register(b0);

struct VertexIn
{
    float3 PosL : POSITION;
    float4 Color : COLOR;
};

struct VertexOut
{
    float4 PosH : SV_POSITION;
    float4 Color : COLOR;
};

VertexOut VS(VertexIn vIn)
{   
    VertexOut vOut;
    vOut.PosH = float4(vIn.PosL, 1.0f);
    vOut.PosH.x += 0.5f * sin(gTime);
    vOut.PosH.y += 0.5f * cos(gTime);
    vOut.Color = vIn.Color;
    return vOut;
}

float4 PS(VertexOut pIn) : SV_TARGET
{
    return pIn.Color;
}