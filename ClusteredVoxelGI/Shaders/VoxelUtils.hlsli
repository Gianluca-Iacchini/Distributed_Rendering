struct GenericMaterial
{
    float4 float4_0;
    float4 float4_1;
    float4 float4_2;
    float4 float4_3;
    
    float float_0;
    float float_1;
    float float_2;
    float float_3;
};

struct VertexIn
{
    float3 PosL : SV_Position;
    float3 NormalL : NORMAL;
    float2 Tex : TEXCOORD;
};

struct VertexOut
{
    float4 PosH : SV_POSITION;
    float4 ShadowPosH : POSITION0;
    float3 PosW : POSITION1;
    float3 NormalW : NORMAL;
    float2 Tex : TEXCOORD;
};

cbuffer Commons : register(b0)
{
    float2 cRenderTargetSize : packoffset(c0);
    float2 cInvRenderTargetSize : packoffset(c0.z);

    float cTotalTime : packoffset(c1);
    float cDeltaTime : packoffset(c1.y);
};

cbuffer Object : register(b1)
{
    float4x4 oWorld : packoffset(c0);
    float4x4 oInvWorld : packoffset(c4);
    float4x4 oTexTransform : packoffset(c8);
    uint oMaterialIndex : packoffset(c12);
};

cbuffer VoxelData : register(b2)
{
    float4x4 vXaxisView : packoffset(c0);
    float4x4 vYaxisView : packoffset(c4);
    float4x4 vZaxisView : packoffset(c8);
    float4x4 vOrthoProj : packoffset(c12);
    
    // Size of the 3D texture
    float3 vVoxelGridDimension : packoffset(c16);
    // Size of a voxel cell
    float3 vVoxelSize : packoffset(c17);
};



SamplerState gSampler : register(s0);


Texture2D gShadowMap : register(t0);

Texture2D gEmissiveTex : register(t1);
Texture2D gNormalMap : register(t2);
Texture2D gDiffuseTex : register(t3);

Texture2D gSpecularTex : register(t4);
Texture2D gAmbientTex : register(t5);
Texture2D gShininessTex : register(t6);
Texture2D gOpacity : register(t7);
Texture2D gBumpMap : register(t8);

StructuredBuffer<GenericMaterial> gMaterials : register(t0, space1);