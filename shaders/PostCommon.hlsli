// PostCommon.hlsli — bindings and helpers shared by the post-process compute
// passes (Resolve.hlsl, Bloom.hlsl).
//
// Design note: every pass here is UAV-only and reads with integer Load, never a
// sampler. That is not a limitation, it is the point. An exact 2x downsample tap
// at the texel centre *is* a 4-texel box average, and an exact 2x upsample tap
// *is* the 4-tap (9,3,3,1)/16 tent, so bilinear filtering would buy nothing.
// Dropping the sampler means no SRVs, which means the bloom mip chain never
// leaves UNORDERED_ACCESS and the only synchronisation between passes is a UAV
// barrier — no per-subresource state tracking anywhere.

#ifndef POST_COMMON_HLSLI
#define POST_COMMON_HLSLI

// Uniform 3-wide binding table across every post pass, so they can all share one
// root signature. u2 is only read by the resolve pass; the bloom passes bind the
// half-res mip there as filler and ignore it.
RWTexture2D<float4> g_postSrc : register(u0);
RWTexture2D<float4> g_postDst : register(u1);
RWTexture2D<float4> g_postAux : register(u2);

cbuffer PostParams : register(b0)
{
    uint2 g_srcDims;
    uint2 g_dstDims;
    float g_ev;              // display EV stops
    float g_bloomThreshold;  // luminance above which a texel blooms
    float g_bloomKnee;       // width of the soft ramp into the threshold
    float g_bloomIntensity;  // 0 disables the bloom composite entirely
};

float PostLuma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

// Normalize an accumulation texel to a mean. The path tracer stores
// xyz = sum of radiance, w = sample count; the denoised staging texture stores
// the already-averaged colour with w = 1, so this is correct for both.
float3 PostAverage(float4 accum)
{
    return (accum.w > 0.0) ? accum.xyz / accum.w : float3(0, 0, 0);
}

// 3x3 tent read, weights (1,2,1)x(1,2,1)/16, clamped at the edges.
//
// This is the smoothing that makes a stack of box-downsampled mips look like
// bloom instead of like a stack of box-downsampled mips. Bilinear interpolation
// alone is not enough: it reconstructs *between* coarse texels but preserves
// their hard boundaries, so the coarsest levels show through the halo as visible
// rectangular steps. Convolving each level with a tent before interpolating is
// what removes them.
float3 PostTent3x3(RWTexture2D<float4> src, uint2 srcDims, int2 c)
{
    const float w[3] = {1.0, 2.0, 1.0};
    int2 hi = int2(srcDims) - 1;
    float3 sum = float3(0, 0, 0);
    [unroll] for (int y = -1; y <= 1; y++)
    {
        [unroll] for (int x = -1; x <= 1; x++)
        {
            int2 p = clamp(c + int2(x, y), int2(0, 0), hi);
            sum += src[uint2(p)].xyz * (w[x + 1] * w[y + 1]);
        }
    }
    return sum * (1.0 / 16.0);
}

// Exact 2x upsample of `src` evaluated at destination texel `dst`.
//
// The fine texel centre (dst + 0.5) maps to coarse coordinate
// (dst + 0.5) / 2 - 0.5, which lands a quarter-texel either side of a coarse
// centre depending on parity, giving interpolation weights of exactly
// (9,3,3,1)/16. Each of the four corners is read through PostTent3x3 rather than
// as a raw texel, which widens the effective kernel enough to hide the mip grid.
// Parity/weight setup shared by both upsample variants.
void PostUpsampleCoords(uint2 dst, out int2 c0, out float2 f)
{
    [unroll] for (int i = 0; i < 2; i++)
    {
        if (dst[i] & 1)
        {
            c0[i] = int(dst[i] >> 1);
            f[i] = 0.25;
        }
        else
        {
            c0[i] = int(dst[i] >> 1) - 1;
            f[i] = 0.75;
        }
    }
}

// Plain 4-tap bilinear 2x upsample, no tent.
//
// Used by the resolve pass only. By the time the chain has finished, mip 0 has
// every coarser level folded into it and is already smooth, so paying 36 loads
// per pixel at full resolution to re-smooth it buys nothing visible — and full
// resolution is where that cost hurts most, since it is the one upsample that
// runs over the entire frame rather than a quarter of it or less.
float3 PostUpsample2xBilinear(RWTexture2D<float4> src, uint2 srcDims, uint2 dst)
{
    int2 c0;
    float2 f;
    PostUpsampleCoords(dst, c0, f);

    int2 hi = int2(srcDims) - 1;
    int2 p0 = clamp(c0, int2(0, 0), hi);
    int2 p1 = clamp(c0 + int2(1, 1), int2(0, 0), hi);

    float3 a = lerp(src[uint2(p0.x, p0.y)].xyz, src[uint2(p1.x, p0.y)].xyz, f.x);
    float3 b = lerp(src[uint2(p0.x, p1.y)].xyz, src[uint2(p1.x, p1.y)].xyz, f.x);
    return lerp(a, b, f.y);
}

float3 PostUpsample2x(RWTexture2D<float4> src, uint2 srcDims, uint2 dst)
{
    int2 c0;
    float2 f;
    PostUpsampleCoords(dst, c0, f);

    float3 s00 = PostTent3x3(src, srcDims, c0 + int2(0, 0));
    float3 s10 = PostTent3x3(src, srcDims, c0 + int2(1, 0));
    float3 s01 = PostTent3x3(src, srcDims, c0 + int2(0, 1));
    float3 s11 = PostTent3x3(src, srcDims, c0 + int2(1, 1));

    float3 a = lerp(s00, s10, f.x);
    float3 b = lerp(s01, s11, f.x);
    return lerp(a, b, f.y);
}

#endif // POST_COMMON_HLSLI
