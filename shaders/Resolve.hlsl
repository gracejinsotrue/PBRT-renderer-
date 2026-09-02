// Resolve.hlsl — display resolve pass.
//
// Converts an HDR accumulation buffer into the display-referred RGBA8 image
// that gets blitted to the backbuffer: normalize by sample count, apply
// exposure compensation, ACES filmic tonemap, gamma 2.2.
//
// This used to live at the tail of RayGen. It was split out so that the tonemap
// curve exists in exactly one place, and so a future bloom pass has somewhere to
// insert itself between the HDR average and the tonemap (bloom is a
// neighbourhood operation and cannot run inside a raygen thread).
//
// Both the live render and the denoised preview go through this same pass. The
// only thing that differs is which HDR texture is bound as g_hdrSource:
//
//   live      g_accum            xyz = sum of radiance, w = sample count
//   denoised  g_denoisedHdr      xyz = OIDN output,     w = 1
//
// Because the source is always normalized by .w, the denoised path needs no
// special case here — it just writes w = 1 when uploading.

RWTexture2D<float4> g_hdrSource : register(u0);
RWTexture2D<float4> g_ldrTarget : register(u1);

cbuffer ResolveParams : register(b0)
{
    uint2 g_dims;
    float g_evCompensation; // display EV stops
    float _resolvePad;
};

[numthreads(8, 8, 1)] void CSResolve(uint3 tid : SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    if (pixel.x >= g_dims.x || pixel.y >= g_dims.y)
        return;

    float4 accum = g_hdrSource[pixel];
    float3 c = (accum.w > 0.0) ? accum.xyz / accum.w : float3(0, 0, 0);

    c = c * pow(2.0, g_evCompensation); // exposure compensation in EV stops (0 = no change)

    // ACES filmic tonemapper (Hill 2016 approximation).
    // Preserves saturation and contrast in midtones better than Reinhard.
    // Input is assumed to be in scene-linear AP1-ish space.
    float a = 2.51, b = 0.03, c0 = 2.43, d = 0.59, e = 0.14;
    c = saturate((c * (a * c + b)) / (c * (c0 * c + d) + e));

    c = pow(c, 1.0 / 2.2); // gamma 2.2
    g_ldrTarget[pixel] = float4(c, 1.0);
}
