// WavefrontPT.hlsl
//
// Persistent-thread path tracer: the second driver for PathStep.
//
// Why this exists
// ---------------
// The megakernel gives one thread one path and runs it to completion. A wave
// therefore runs its bounce loop as many times as its *longest* path needs,
// while every lane whose path already died sits idle. PROFILE_WAVES measures
// exactly that waste: utilisation is 56% on cbox, 50% on pool_store, 37.6% on
// the volume scene, so 1.6x to 2.7x of the lane-iterations are thrown away.
//
// The fix is not to make paths shorter but to stop a dead lane from waiting. A
// fixed pool of threads pulls pixels off a global queue, and the inner loop
// advances every lane by exactly one bounce per iteration -- on whatever path
// that lane happens to be holding. A lane whose path finishes retires it and
// immediately pulls the next pixel, so the wave stays full until the queue runs
// dry, and lanes at bounce 0 and bounce 20 coexist happily in the same wave.
//
// This is Aila & Laine's persistent-threads formulation rather than the
// classical wavefront tracer in the roadmap. It buys the same thing for far less
// machinery: no per-stage global path buffers, no multi-pass dispatch, no
// compaction kernel. Path state stays in registers exactly as it does today.
// What it does NOT do is fix material divergence -- lanes in one wave still run
// different BSDF branches, which the same measurement suggests is the larger
// half of the budget and needs sorting to reach.
//
// Requires inline ray tracing: PathStep calls TraceRay when USE_RAYQUERY=0, and
// TraceRay does not exist in a compute shader. USE_RAYQUERY defaults to 1.

#include "Common.hlsli"
#include "RNG.hlsli"
#include "GeometryUtils.hlsli"
#include "RayQueryTrace.hlsli"
#include "Microfacet.hlsli"
#include "Disney.hlsli"
#include "Hair.hlsli"
#include "Material.hlsli"
#include "Emitter.hlsli"
#include "Envmap.hlsli"
#include "Volume.hlsl"
#include "Subsurface.hlsli"
#include "PathTrace.hlsli"

#if !USE_RAYQUERY
#error "WavefrontPT requires USE_RAYQUERY=1: TraceRay is not available in a compute shader."
#endif

// Work queue: a single counter handing out pixel indices. One dword; the rest of
// the buffer is unused.
RWByteAddressBuffer g_pathQueue : register(u6);

// Queue slots are handed out in 8x8 tiles, not scanline order.
//
// This is not a micro-optimisation, it is what makes the whole approach viable.
// Path regeneration trades divergence for locality: a wave starts with 32
// adjacent pixels, but as its lanes finish at different bounces they refill from
// wherever the counter happens to be, and under scanline order those refills are
// scattered across the image. Every lane in the wave then walks a different part
// of the BVH and samples different texture pages. Measured on pool_store (531k
// triangles, 20+ 1024px textures) scanline order ran 0.54x -- nearly twice as
// slow as the megakernel -- while cbox, which has 2k triangles and no textures,
// was happily 1.50x faster. The divergence win was real; it was just smaller
// than the cache loss.
//
// Tiling keeps consecutive slots spatially compact, so a refilled lane lands
// near its wave-mates instead of across the frame. Only the ragged right/bottom
// edge is wasted, and a wasted slot costs one bounds test.
static const uint kTile = 8;

bool SlotToPixel(uint slot, uint2 dims, out uint2 pixel)
{
    uint tilesX = (dims.x + kTile - 1) / kTile;
    uint perTile = kTile * kTile;
    uint tile = slot / perTile;
    uint within = slot - tile * perTile;
    uint2 origin = uint2((tile % tilesX) * kTile, (tile / tilesX) * kTile);
    pixel = origin + uint2(within % kTile, within / kTile);
    return pixel.x < dims.x && pixel.y < dims.y;
}

uint SlotCount(uint2 dims)
{
    uint tilesX = (dims.x + kTile - 1) / kTile;
    uint tilesY = (dims.y + kTile - 1) / kTile;
    return tilesX * tilesY * kTile * kTile;
}

[numthreads(1, 1, 1)] void CSResetQueue()
{
    // Cheaper and simpler than ClearUnorderedAccessViewUint, which would need a
    // second non-shader-visible descriptor for the same resource.
    g_pathQueue.Store(0, 0);
}

[numthreads(64, 1, 1)] void CSPathTracePersistent(uint3 tid : SV_DispatchThreadID)
{
    uint2 dims;
    g_accum.GetDimensions(dims.x, dims.y);
    uint total = SlotCount(dims);

    PathState P = (PathState)0;
    uint2 pixel = uint2(0, 0);
    float4 accumPrev = float4(0, 0, 0, 0);
    float2 momentsPrev = float2(0, 0);
    bool alive = false;

    [loop] while (true)
    {
        if (!alive)
        {
            // Wave-aggregated fetch: one atomic per wave instead of one per lane.
            // Every lane that needs work contributes to a single InterlockedAdd
            // and then takes its own slot out of the returned block, which cuts
            // contention on the counter by up to the wave width.
            uint need = WaveActiveCountBits(true);
            uint base = 0;
            if (WaveIsFirstLane())
                g_pathQueue.InterlockedAdd(0, need, base);
            base = WaveReadLaneFirst(base);
            uint slot = base + WavePrefixCountBits(true);
            if (slot >= total)
                break; // queue drained

            if (!SlotToPixel(slot, dims, pixel))
                continue; // ragged tile edge, outside the image

            RNG rng;
            if (!BeginPixel(pixel, dims, rng, accumPrev, momentsPrev))
                continue; // converged pixel: costs one queue slot, no rays

            P = InitPathState(GeneratePrimaryRay(pixel, dims, rng), rng);
            alive = true;
        }

        // Exactly the step TracePath performs, so a pixel traced here draws the
        // same sample as the megakernel would and the two images match.
        P.bounce = (int)P.pathLen;
        P.pathLen++;
        alive = PathStep(P, pixel, dims);
        if (alive && P.pathLen >= (uint)MAX_BOUNCES)
            alive = false;

        if (!alive)
            WritePathResult(pixel, P, accumPrev, momentsPrev);
    }
}
