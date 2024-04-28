float gTime : register(b0);
Texture2D gTex : register(t0);

SamplerState gSampler : register(s0);

struct VertexIn
{
    float3 PosL : POSITION;
    float2 Tex : TEXCOORD;
    float4 Color : COLOR;
};

struct VertexOut
{
    float4 PosH : SV_POSITION;
    float2 Tex : TEXCOORD;
    float4 Color : COLOR;
};

VertexOut VS(VertexIn vIn)
{   
    VertexOut vOut;
    vOut.PosH = float4(vIn.PosL, 1.0f);
    vOut.PosH.x += 0.5f * sin(gTime);
    vOut.PosH.y += 0.5f * cos(gTime);
    vOut.Tex = vIn.Tex;
    vOut.Color = vIn.Color;
    return vOut;
}

float4 PS(VertexOut pIn) : SV_TARGET
{
    //return float4(pIn.Tex, 1.0f, 1.0f);
    return gTex.Sample(gSampler, pIn.Tex);
}