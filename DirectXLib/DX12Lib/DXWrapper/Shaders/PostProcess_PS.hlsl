#include "Common.hlsli"

cbuffer cbCommons : register(b0)
{
    Commons commons;
};

cbuffer cbCamera : register(b1)
{
    Camera camera;
}

Texture2D gBufferaWorld : register(t0);
Texture2D gBufferNormal : register(t1);
Texture2D gBufferDiffuse : register(t2);
Texture2D gBufferMetallicRoughnessAO : register(t3);
Texture2D gDeferredResult : register(t4);
Texture2D gRadianceTexture : register(t5);

float GaussianWeight(float x, float y, float sigma)
{
    return exp(-((x * x + y * y) / (2.0f * sigma * sigma))) / (2.0f * 3.14159f * sigma * sigma);
}

float4 PS(VertexOutPosTex pIn) : SV_Target
{
    
    float4 diffuse = gBufferDiffuse.Sample(gSampler, pIn.Tex);
    float4 RMA = gBufferMetallicRoughnessAO.Sample(gSampler, pIn.Tex);
        
    // w coordinate is used to keep track of geometry. If there is no geometry, the w value is 1.0f so we can discard the pixel
    if (RMA.w >= 1.0f)
        discard;
    
    // glTF stores roguhness in the G channel, metallic in the B channel and AO in the R channel
    float roughness = RMA.g;
    float metallic = RMA.b;
    float occlusion = RMA.r;

    float3 cDiff = lerp(diffuse.rgb, float3(0, 0, 0), metallic) * occlusion;
    
    
    float4 deferredRes = gDeferredResult.Sample(gSampler, pIn.Tex);
    float3 radiance = float3(0.0f, 0.0f, 0.0f);
    
    uint width, height;
    gRadianceTexture.GetDimensions(width, height);

    float2 texelSize = 1.0 / float2(width, height);

// Bilateral filter parameters
    float sigmaSpatial = 20.0f; // Controls spatial smoothing (e.g., pixel distance)
    float sigmaIntensity = 12.0f; // Controls intensity smoothing (e.g., color difference)
    float worldThreshold = 2.0f; // Maximum allowed difference in world coordinates

    float4 centerValue = gRadianceTexture.Sample(gSampler, pIn.Tex); // Current pixel value
    
    if (centerValue.w > 0.0f)
    {
        float3 centerWorldCoord = gBufferaWorld.Sample(gSampler, pIn.Tex).xyz; // Center fragment world coordinate

        float weightSum = 0.0;

        // Apply bilateral filter in a 3x3 neighborhood
        for (int y = -3; y <= 3; y++)
        {
            for (int x = -3; x <= 3; x++)
            {
                // Neighbor UV coordinates
                float2 neighborUV = pIn.Tex + float2(x, y) * texelSize;

                // Sample the neighbor values
                float3 neighborValue = gRadianceTexture.SampleLevel(gSampler, neighborUV, 0).xyz;
                float3 neighborWorldCoord = gBufferaWorld.SampleLevel(gSampler, neighborUV, 0).xyz;

                // Compute the difference in world coordinates
                float3 worldCoordDiff = neighborWorldCoord - centerWorldCoord;

                // Skip this neighbor if the world coordinate difference exceeds the threshold
                if (length(worldCoordDiff) > worldThreshold)
                    continue;

                // Compute spatial weight (Gaussian based on distance)
                float spatialWeight = exp(-float(x * x + y * y) / (2.0f * sigmaSpatial * sigmaSpatial));

                // Compute intensity weight (Gaussian based on color difference)
                float intensityWeight = exp(-dot(centerValue.xyz - neighborValue, centerValue.xyz - neighborValue) / (2.0f * sigmaIntensity * sigmaIntensity));

                // Combine weights
                float weight = spatialWeight * intensityWeight;

                // Accumulate weighted value
                radiance += neighborValue * weight;
                weightSum += weight;
            }
        }

        // Normalize the smoothed value
        radiance /= weightSum;
    }
    else
    {
        radiance = centerValue.xyz;
    }
    
    float4 result = deferredRes;
    result.xyz += cDiff * radiance * 0.8f;
    
    return result;
}