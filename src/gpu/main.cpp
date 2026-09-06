#include "Win32Application.h"
#include "DXRApp.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: nori-dxr <scene.xml> [--headless] [--denoise]\n");
        fprintf(stderr, "  --headless  : render and exit (respects sampler sampleCount)\n");
        fprintf(stderr, "  --denoise   : also run OIDN on the result, save snapshot_N_denoised.exr\n");
        fprintf(stderr, "  --png       : also save the tonemapped display image (exposure/bloom/ACES) as PNG\n");
        fprintf(stderr, "  --profile   : GPU-timestamp the DispatchRays call, print ms/frame stats\n");
        fprintf(stderr, "  --novsync   : present without waiting for vblank (windowed; pair with --profile)\n");
        fprintf(stderr, "  --clamp N   : clamp indirect contributions to luminance N (biased; default off)\n");
        fprintf(stderr, "  --adaptive T[,W] : stop refining a pixel once its relative standard\n");
        fprintf(stderr, "  --restir R[,K] : ReSTIR DI spatial reuse, radius R px, K neighbours\n");
        fprintf(stderr, "                   (no-op unless the shader was built -D USE_RIS=1)\n");
        fprintf(stderr, "                     error falls below T, after a W-sample warm-up (default 32)\n");
        fprintf(stderr, "Example: nori-dxr ..\\scenes\\a4\\cbox\\cbox_mis.xml\n");
        return 1;
    }

    bool headless = false;
    bool denoise = false;
    bool png = false;
    bool profile = false;
    bool vsync = true;
    float clamp = 0.0f;    // 0 = firefly clamp disabled
    float adaptive = 0.0f; // 0 = adaptive sampling disabled
    uint32_t adaptiveWarmup = 32;
    float restirRadius = 0.0f; // 0 = ReSTIR spatial reuse disabled
    uint32_t restirNeighbours = 4;
    for (int i = 2; i < argc; i++)
    {
        if (strcmp(argv[i], "--novsync") == 0)
        {
            vsync = false;
        }
        else if (strcmp(argv[i], "--headless") == 0)
        {
            headless = true;
        }
        else if (strcmp(argv[i], "--denoise") == 0)
        {
            denoise = true;
        }
        else if (strcmp(argv[i], "--png") == 0)
        {
            png = true;
        }
        else if (strcmp(argv[i], "--profile") == 0)
        {
            profile = true;
        }
        else if (strcmp(argv[i], "--clamp") == 0 && i + 1 < argc)
        {
            clamp = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--restir") == 0 && i + 1 < argc)
        {
            // "R" or "R,K": neighbour radius in pixels, optional neighbour count.
            const char *arg = argv[++i];
            restirRadius = (float)atof(arg);
            if (const char *comma = strchr(arg, ','))
                restirNeighbours = (uint32_t)atoi(comma + 1);
            restirNeighbours = restirNeighbours > 8u ? 8u : restirNeighbours;
        }
        else if (strcmp(argv[i], "--adaptive") == 0 && i + 1 < argc)
        {
            // "T" or "T,W": threshold, then an optional warm-up sample count.
            const char *arg = argv[++i];
            adaptive = (float)atof(arg);
            if (const char *comma = strchr(arg, ','))
                adaptiveWarmup = (uint32_t)atoi(comma + 1);
            if (adaptiveWarmup < 2)
            {
                fprintf(stderr, "[adaptive] warm-up raised to 2 (a variance estimate needs two samples)\n");
                adaptiveWarmup = 2;
            }
        }
    }

    try
    {
        HINSTANCE hInstance = GetModuleHandle(nullptr);
        DXRApp app(argv[1], headless);
        app.SetProfiling(profile);
        app.SetFireflyClamp(clamp);
        app.SetAdaptive(adaptive, adaptiveWarmup);
        app.SetReSTIR(restirRadius, restirNeighbours);
        app.SetVSync(vsync); // before OnInit: CreateSwapChain needs the tearing flag

        if (headless)
        {
            // headless mode: init, render to target samples, save, exit
            app.OnInit();
            fprintf(stderr, "[headless] Rendering...\n");

            // Render until target sample count is reached
            uint32_t target = app.GetTargetSamples();
            if (target == 0)
            {
                fprintf(stderr, "[headless] ERROR: sampler sampleCount not set in scene\n");
                app.OnDestroy();
                return 1;
            }
            auto _perfT0 = std::chrono::high_resolution_clock::now();
            for (uint32_t frame = 0; frame < 100000; frame++)
            {
                app.OnRender();
                uint32_t current = app.GetFrameCount();
                if (frame % 200 == 0)
                {
                    fprintf(stderr, "[%u/%u] ", current, target);
                    fflush(stderr);
                }
                if (current >= target)
                {
                    fprintf(stderr, "\n");
                    break;
                }
            }
            {
                double _ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::high_resolution_clock::now() - _perfT0)
                                 .count();
                fprintf(stderr, "[perf] render %u spp in %.1f ms = %.3f ms/spp (%.1f spp/s)\n",
                        target, _ms, _ms / (double)target, (double)target * 1000.0 / _ms);
            }

            app.ReportAdaptiveStats();
            fprintf(stderr, "[headless] Saving EXR...\n");
            app.SaveSnapshotEXR();
            // The EXR is scene-linear radiance and deliberately carries no
            // bloom; the PNG is the display-referred image, where exposure,
            // bloom, ACES and gamma have been applied.
            if (png)
                app.SaveSnapshotPNG(false);
            if (denoise)
            {
                fprintf(stderr, "[headless] Denoising (OIDN)...\n");
                app.DenoiseAndSaveEXR();
                if (png)
                {
                    app.DenoiseToViewport(); // stages the HDR, blooms, resolves
                    app.SaveSnapshotPNG(true);
                }
            }
            app.OnDestroy();
            fprintf(stderr, "[headless] Done.\n");
            return 0;
        }
        else
        {
            return Win32Application::Run(&app, hInstance, SW_SHOW);
        }
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "[FATAL] %s\n", e.what());
        return 1;
    }
}
