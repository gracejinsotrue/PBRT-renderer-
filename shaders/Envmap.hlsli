// Envmap.hlsli. Equirectangular environment map evaluation and importance sampling via a 2D CDF
//
// Requires: Common.hlsli for resource bindings.

#ifndef ENVMAP_HLSLI
#define ENVMAP_HLSLI

#include "Common.hlsli"

// ============================================================================
// Evaluation
// ============================================================================

// Evaluate equirectangular envmap for a world-space direction
float3 EvalEnvmap(float3 dir)
{
    float3 d = normalize(dir);
    float phi = atan2(d.z, d.x);
    if (phi < 0.0)
        phi += 2.0 * M_PI;
    float theta = acos(clamp(d.y, -1.0, 1.0));
    float2 uv = float2(phi / (2.0 * M_PI), theta / M_PI);
    return g_envmap.SampleLevel(g_envmapSampler, uv, 0).rgb * envmapScale;
}

// ============================================================================
// CDF search, shared by sampling and pdf lookup
// ============================================================================

// Binary search over a monotonic CDF stored in a ByteAddressBuffer.
uint EnvmapCdfSearch(ByteAddressBuffer buf, uint byteOffset, uint numEntries, float u)
{
    uint lo = 0;
    uint hi = numEntries - 1;
    while (lo < hi)
    {
        uint mid = (lo + hi) / 2;
        float val = asfloat(buf.Load(byteOffset + mid * 4));
        if (val <= u)
            lo = mid + 1;
        else
            hi = mid;
    }
    return max(0, int(lo) - 1);
}

// ============================================================================
// Importance sampling
// ============================================================================

// Sample a direction from the environment map proportional to radiance x sin(theta).
// u1, u2 are independent uniform samples in [0, 1).
// Outputs:
//   dir      - world-space unit direction toward the sampled pixel center
//   radiance - envmap radiance along dir
//   pdf      - probability density in solid-angle per steradian
// todo: the direction is built from the pixel center; jittering should be added later
void SampleEnvmap(float u1, float u2, out float3 dir, out float3 radiance, out float pdf)
{
    uint W, H;
    g_envmap.GetDimensions(W, H);

    // 1. pick a row via the marginal CDF (H + 1 entries).
    uint y = EnvmapCdfSearch(g_envmapMarginalCdf, 0, H + 1, u1);
    // recover within-row jitter v ∈ [0,1) from u1's position inside the row's CDF cell.
    float marg_y0 = asfloat(g_envmapMarginalCdf.Load(y * 4));
    float marg_y1 = asfloat(g_envmapMarginalCdf.Load((y + 1) * 4));
    float pMar = marg_y1 - marg_y0;
    float jitter_v = (pMar > 1e-20) ? saturate((u1 - marg_y0) / pMar) : 0.5;

    // 2. pick a column within that row via the conditional CDF, where row y starts at float offset y * (W + 1), meaning byte offset y*(W+1)*4.
    uint rowBytes = (W + 1) * 4;
    uint x = EnvmapCdfSearch(g_envmapConditionalCdf, y * rowBytes, W + 1, u2);

    // recover within-column jitter u ∈ [0,1) from u2's position inside the column's CDF cell.
    float cond_x0 = asfloat(g_envmapConditionalCdf.Load(y * rowBytes + x * 4));
    float cond_x1 = asfloat(g_envmapConditionalCdf.Load(y * rowBytes + (x + 1) * 4));
    float pCon = cond_x1 - cond_x0;
    float jitter_u = (pCon > 1e-20) ? saturate((u2 - cond_x0) / pCon) : 0.5;

    // 3. Compute the direction at the jittered sub-pixel position.
    float u = ((float)x + jitter_u) / (float)W;
    float v = ((float)y + jitter_v) / (float)H;
    float phi = 2.0 * M_PI * u;
    float theta = M_PI * v;
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);
    dir = float3(sinTheta * cos(phi),
                 cosTheta,
                 sinTheta * sin(phi));

    // 4. compute radiance by sampling the envmap texture at the jittered UV (bilinear).
    radiance = g_envmap.SampleLevel(g_envmapSampler, float2(u, v), 0).rgb * envmapScale;

    // 5. pixel-space pdf p_pixel = p_marginal(y) * p_conditional(x | y).
    float pPixel = pMar * pCon;

    // do the jacobian pixel-space -> direction-space: p_omega = p_pixel * (W * H) / (2 * pi^2 * sin(theta)).
    float denom = 2.0 * M_PI * M_PI * max(sinTheta, 1e-8);
    pdf = pPixel * (float)(W * H) / denom;
}

// ============================================================================
// PDF query
// ============================================================================

// Compute the solid-angle pdf that SampleEnvmap would have produced for the given world-space direction.  Used for MIS weights when the BSDF/path
// sampler picks a direction that happens to hit the envmap.
float EnvmapPdfDirection(float3 dir)
{
    uint W, H;
    g_envmap.GetDimensions(W, H);

    float3 d = normalize(dir);
    float phi = atan2(d.z, d.x);
    if (phi < 0.0)
        phi += 2.0 * M_PI;
    float theta = acos(clamp(d.y, -1.0, 1.0));
    float sinTheta = sin(theta);

    float u = phi / (2.0 * M_PI);
    float v = theta / M_PI;
    uint x = min((uint)(u * (float)W), W - 1);
    uint y = min((uint)(v * (float)H), H - 1);

    uint rowBytes = (W + 1) * 4;
    float pMar = asfloat(g_envmapMarginalCdf.Load((y + 1) * 4)) -
                 asfloat(g_envmapMarginalCdf.Load(y * 4));
    float pCon = asfloat(g_envmapConditionalCdf.Load(y * rowBytes + (x + 1) * 4)) -
                 asfloat(g_envmapConditionalCdf.Load(y * rowBytes + x * 4));
    float pPixel = pMar * pCon;

    float denom = 2.0 * M_PI * M_PI * max(sinTheta, 1e-8);
    return pPixel * (float)(W * H) / denom;
}

#endif // ENVMAP_HLSLI
