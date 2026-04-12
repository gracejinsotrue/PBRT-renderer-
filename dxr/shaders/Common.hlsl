// Common bindings and structures shared across all ray tracing shaders.

// The top-level acceleration structure, bound as SRV in register t0.
RaytracingAccelerationStructure g_scene : register(t0);

// Output UAV texture, which DispatchRays writes to currently, then we copy to the swap chain.
RWTexture2D<float4> g_output : register(u0);

// Per-frame constants pushed via root constants.
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
