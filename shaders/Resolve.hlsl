// Resolve.hlsl — display resolve pass.
//
// Converts an HDR accumulation buffer into the display-referred RGBA8 image
// that gets blitted to the backbuffer: normalize by sample count, apply
// exposure compensation, composite bloom, ACES filmic tonemap, gamma 2.2.
//
// This used to live at the tail of RayGen. It was split out so that the tonemap
// curve exists in exactly one place, and so bloom has somewhere to insert
// itself between the HDR average and the tonemap. That ordering is not
// negotiable: ACES saturates, so bloom applied after it would have nothing
// bright left to work with.
//
// Both the live render and the denoised preview go through this same pass. The
// only thing that differs is which HDR texture is bound as g_postSrc:
//
//   live      g_accum            xyz = sum of radiance, w = sample count
//   denoised  g_denoisedHdr      xyz = OIDN output,     w = 1
//
// Because the source is always normalized by .w, the denoised path needs no
// special case here — it just writes w = 1 when uploading.

#include "PostCommon.hlsli"

[numthreads(8, 8, 1)] void CSResolve(uint3 tid : SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    if (pixel.x >= g_dstDims.x || pixel.y >= g_dstDims.y)
        return;

    float3 c = PostAverage(g_postSrc[pixel]) * exp2(g_ev);

    // Bloom lives at half resolution in g_postAux, so it is tented back up to
    // full res rather than point-sampled — nearest here would show as blocky
    // stair-stepping along the edge of every glow.
    if (g_bloomIntensity > 0.0)
        c += PostUpsample2xBilinear(g_postAux, g_dstDims / 2, pixel) * g_bloomIntensity;

    // ACES filmic tonemapper (Hill 2016 approximation).
    // Preserves saturation and contrast in midtones better than Reinhard.
    // Input is assumed to be in scene-linear AP1-ish space.
    float a = 2.51, b = 0.03, c0 = 2.43, d = 0.59, e = 0.14;
    c = saturate((c * (a * c + b)) / (c * (c0 * c + d) + e));

    c = pow(c, 1.0 / 2.2); // gamma 2.2
    g_postDst[pixel] = float4(c, 1.0);
}
