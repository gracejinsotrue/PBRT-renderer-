// Ray Generation Shader
// Runs once per pixel. Computes a primary ray from the camera and calls TraceRay.

#include "Common.hlsl"

[shader("raygeneration")] void RayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dims = DispatchRaysDimensions().xy;

    // Map pixel to [0,1] UV coordinates.
    float2 uv = (float2(pixel) + 0.5f) / float2(dims);

    // Compute ray direction from camera image plane.
    float3 origin = camPos;
    float3 direction = normalize(
        camLowerLeftCorner + uv.x * camHorizontal + uv.y * camVertical - origin);

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.001f;
    ray.TMax = 1e20f;

    RayPayload payload;
    payload.color = float3(0, 0, 0);

    // trace the ray against the TLAS.
    // Flags: none
    // InstanceInclusionMask: 0xFF
    // HitGroupIndex: 0 (first hit group in the shader table)
    // MissShaderIndex: 0 (first miss shader in the shader table)
    TraceRay(
        g_scene,       // acceleration structure
        RAY_FLAG_NONE, // ray flags
        0xFF,          // instance mask
        0,             // hit group index offset
        0,             // hit group geometry stride
        0,             // miss shader index
        ray,           // ray descriptor
        payload);

    g_output[pixel] = float4(payload.color, 1.0f);
}
