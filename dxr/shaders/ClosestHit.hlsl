// Closest Hit Shader, invoked when a ray finds its nearest intersection.

#include "Common.hlsl"

[shader("closesthit")] void ClosestHit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    // DXR gives barycentrics (u, v) for the hit triangle, but the third barycentric coordinate is w = 1 - u - v.
    float3 barycentrics = float3(
        1.0f - attr.barycentrics.x - attr.barycentrics.y,
        attr.barycentrics.x,
        attr.barycentrics.y);

    payload.color = barycentrics;
}
