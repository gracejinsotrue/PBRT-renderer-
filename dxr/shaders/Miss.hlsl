// Miss Shader
// Invoked when a ray doesn't hit any geometry.
// returns the default screen color for rays that miss geometry

#include "Common.hlsl"

[shader("miss")] void Miss(inout RayPayload payload)
{
    float t = 0.5f * (WorldRayDirection().y + 1.0f);
    payload.color = lerp(float3(1, 1, 1), float3(0.5f, 0.7f, 1.0f), t);
}
