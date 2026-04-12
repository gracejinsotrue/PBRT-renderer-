
RaytracingAccelerationStructure g_scene : register(t0);
RWTexture2D<float4> g_output : register(u0);

cbuffer CameraParams : register(b0)
{
    float3 camPos;
    float pad0;
    float3 camLowerLeftCorner;
    float pad1;
    float3 camHorizontal;
    float pad2;
    float3 camVertical;
    float pad3;
};

struct RayPayload
{
    float3 color;
};

[shader("raygeneration")] void RayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dims = DispatchRaysDimensions().xy;

    float2 uv = (float2(pixel) + 0.5f) / float2(dims);
    uv.y = 1.0f - uv.y;

    float3 origin = camPos;
    float3 direction = normalize(
        camLowerLeftCorner + uv.x * camHorizontal + uv.y * camVertical - origin);

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.001f;
    ray.TMax = 100.0f;

    RayPayload payload;
    payload.color = float3(0, 0, 0);

    TraceRay(g_scene, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);
    g_output[pixel] = float4(payload.color, 1.0f);
}

    [shader("closesthit")] void ClosestHit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    // todo: rewrite the li() integrator instead of depth shading
    float depth = RayTCurrent();
    float shade = saturate(1.0 - depth / 6.0);

    uint id = InstanceID();
    float r = float((id * 67u + 13u) % 256u) / 255.0;
    float g = float((id * 131u + 77u) % 256u) / 255.0;
    float b = float((id * 211u + 173u) % 256u) / 255.0;
    float3 tint = normalize(float3(r, g, b) + 0.3);

    payload.color = tint * shade;
}

[shader("miss")] void Miss(inout RayPayload payload)
{
    float t = 0.5f * (WorldRayDirection().y + 1.0f);
    payload.color = lerp(float3(0.1, 0.1, 0.1), float3(0.2, 0.3, 0.5), t);
}
