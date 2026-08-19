// RNG.hlsli — progressive sampler.
//
// Owen-scrambled Sobol' after Burley 2020, "Practical Hash-Based Owen Scrambling". This replaces the previous PCG white-noise stream to lower variance per sample
// for a faster convergence.

#ifndef RNG_HLSLI
#define RNG_HLSLI

uint PCGHash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Sobol' base-2, MSB-aligned fixed point. Dimension 0 is the van der Corput; dimension 1 uses the Gray-code direction numbers.
uint Sobol2D(uint index, uint axis)
{
    if (axis == 0u)
        return reversebits(index);
    uint r = 0u;
    uint dirn = 1u << 31;
    for (uint i = index; i != 0u; i >>= 1u)
    {
        if (i & 1u)
            r ^= dirn;
        dirn ^= dirn >> 1u;
    }
    return r;
}

// Hash-based Owen scramble, Laine & Karras / Burley 2020.
uint LaineKarras(uint x, uint seed)
{
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return x;
}

uint OwenScramble(uint x, uint seed)
{
    x = reversebits(x);
    x = LaineKarras(x, seed);
    x = reversebits(x);
    return x;
}

struct RNG
{
    uint pixelSeed;   // fixed per-pixel scramble seed
    uint sampleIndex; // progressive sample number, same as fram ecount
    uint dim;         // running path-dimension counter
    uint state;       // payload-threaded legacy PCG stream for the any-hit alpha test
    uint pcgWhite;    // white-noise stream, only used when USE_SOBOL=0
};

RNG InitRNG(uint2 pixel, uint2 dims, uint frame)
{
    RNG rng;
    uint pid = pixel.y * dims.x + pixel.x;
    rng.pixelSeed = PCGHash(pid ^ 0x9e3779b9u); // depends on pixel only
    rng.sampleIndex = frame;
    rng.dim = 0u;
    rng.state = PCGHash(pid + frame * dims.x * dims.y); // unchanged any-hit seed
    rng.pcgWhite = PCGHash(pid * 0x9e3779b9u + frame * 0x85ebca6bu + 1u);
    return rng;
}

// Sampler selection for convergence A/B. Default is the Owen-scrambled Sobol'
// sequence; -D USE_SOBOL=0 restores the original PCG white-noise stream so the
// two can be compared at equal spp against a converged reference. Both are
// unbiased, so they converge to the same image -- the difference is the *rate*.
#ifndef USE_SOBOL
#define USE_SOBOL 1
#endif

float NextFloat(inout RNG rng)
{
#if !USE_SOBOL
    // Original white-noise path: one PCG step per dimension, decorrelated per
    // pixel and per sample by the seed built in InitRNG.
    rng.dim++;
    rng.pcgWhite = PCGHash(rng.pcgWhite);
    return min(float(rng.pcgWhite) * 2.3283064365386963e-10, 0.99999994);
#else
    uint d = rng.dim++;
    uint axis = d & 1u;
    uint pad = d >> 1u;
    // Shuffle the Sobol index per pad so different dimension-pairs use different
    // points then Owen-scramble the value per pixel, pad, axis
    uint sIdx = OwenScramble(rng.sampleIndex, PCGHash(pad * 0x9e3779b9u + 1u));
    uint v = OwenScramble(Sobol2D(sIdx, axis),
                          PCGHash(rng.pixelSeed + pad * 0x6c8e9cf5u + axis * 0x68bc21ebu));
    return min(float(v) * 2.3283064365386963e-10, 0.99999994); // [0,1)
#endif
}

#endif // RNG_HLSLI
