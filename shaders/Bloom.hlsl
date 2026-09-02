// Bloom.hlsl — HDR bloom mip chain.
//
// Three passes, dispatched as: prefilter once, downsample (M-1) times walking
// down the chain, then upsample (M-1) times walking back up and accumulating
// into each finer mip. Mip 0 is half the render resolution; the resolve pass
// reads it back through the same 2x tent used here.
//
// The firefly problem
// -------------------
// Most bloom implementations blur a rasterized image, where the input is clean.
// Here the input is a Monte Carlo mean with the firefly clamp effectively
// disabled (kFireflyClamp is FLT_MAX), so at low sample counts a single unlucky
// path can leave a 1000x outlier in one texel. A naive bright-pass would latch
// onto it and the blur would smear it into a large soft halo that slowly
// deflates as the estimate converges — the image would visibly breathe.
//
// The prefilter defends against this twice: a Karis average (weight each tap by
// 1/(1+luma)) so one hot texel cannot dominate its 2x2 block, and a soft-knee
// threshold so texels ramp across the cutoff instead of popping.

#include "PostCommon.hlsli"

// Karis-weighted 2x2 downsample of the HDR average, then soft-knee bright pass.
[numthreads(8, 8, 1)] void CSBloomPrefilter(uint3 tid : SV_DispatchThreadID)
{
    uint2 dst = tid.xy;
    if (dst.x >= g_dstDims.x || dst.y >= g_dstDims.y)
        return;

    uint2 base = dst * 2;
    uint2 hi = g_srcDims - uint2(1, 1);
    float exposure = exp2(g_ev);

    float3 sum = float3(0, 0, 0);
    float wsum = 0.0;
    [unroll] for (int y = 0; y < 2; y++)
    {
        [unroll] for (int x = 0; x < 2; x++)
        {
            uint2 p = min(base + uint2(x, y), hi);
            float3 c = PostAverage(g_postSrc[p]) * exposure;
            // Karis: firefly suppression. A texel 1000x brighter than its
            // neighbours contributes at ~1/1000 the weight instead of 1000x.
            float w = 1.0 / (1.0 + PostLuma(c));
            sum += c * w;
            wsum += w;
        }
    }
    float3 c = sum / max(wsum, 1e-6);

    // Soft-knee threshold. Below (threshold - knee) nothing blooms; across the
    // 2*knee band the response ramps quadratically; above it the curve is the
    // plain (brightness - threshold) bright pass.
    float br = max(c.r, max(c.g, c.b));
    float soft = clamp(br - g_bloomThreshold + g_bloomKnee, 0.0, 2.0 * g_bloomKnee);
    soft = soft * soft / (4.0 * g_bloomKnee + 1e-6);
    float contrib = max(soft, br - g_bloomThreshold) / max(br, 1e-6);

    float3 outc = c * contrib;
    if (any(isnan(outc)) || any(isinf(outc)))
        outc = float3(0, 0, 0);
    g_postDst[dst] = float4(outc, 1.0);
}

// Exact 2x box downsample, mip N -> mip N+1.
[numthreads(8, 8, 1)] void CSBloomDownsample(uint3 tid : SV_DispatchThreadID)
{
    uint2 dst = tid.xy;
    if (dst.x >= g_dstDims.x || dst.y >= g_dstDims.y)
        return;

    uint2 base = dst * 2;
    uint2 hi = g_srcDims - uint2(1, 1);
    float3 sum = float3(0, 0, 0);
    [unroll] for (int y = 0; y < 2; y++)
        [unroll] for (int x = 0; x < 2; x++)
            sum += g_postSrc[min(base + uint2(x, y), hi)].xyz;

    g_postDst[dst] = float4(sum * 0.25, 1.0);
}

// Tent upsample of the coarser mip, accumulated into the finer one. Summing
// every level on the way back up is what turns a stack of box blurs into the
// wide, smooth falloff bloom is supposed to have.
[numthreads(8, 8, 1)] void CSBloomUpsample(uint3 tid : SV_DispatchThreadID)
{
    uint2 dst = tid.xy;
    if (dst.x >= g_dstDims.x || dst.y >= g_dstDims.y)
        return;

    float3 up = PostUpsample2x(g_postSrc, g_srcDims, dst);
    float3 prev = g_postDst[dst].xyz;
    g_postDst[dst] = float4(prev + up, 1.0);
}
