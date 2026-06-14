#include "Win32Application.h"
#include "DXRApp.h"
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: nori-dxr <scene.xml> [--headless] [--denoise]\n");
        fprintf(stderr, "  --headless  : render and exit (respects sampler sampleCount)\n");
        fprintf(stderr, "  --denoise   : also run OIDN on the result, save snapshot_N_denoised.exr\n");
        fprintf(stderr, "Example: nori-dxr ..\\scenes\\a4\\cbox\\cbox_mis.xml\n");
        return 1;
    }

    bool headless = false;
    bool denoise = false;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--headless") == 0) {
            headless = true;
        } else if (strcmp(argv[i], "--denoise") == 0) {
            denoise = true;
        }
    }

    try {
        HINSTANCE hInstance = GetModuleHandle(nullptr);
        DXRApp app(argv[1], headless);

        if (headless) {
            // Headless mode: init, render to target samples, save, exit
            app.OnInit();
            fprintf(stderr, "[headless] Rendering...\n");

            // Render until target sample count is reached
            uint32_t target = app.GetTargetSamples();
            if (target == 0) {
                fprintf(stderr, "[headless] ERROR: sampler sampleCount not set in scene\n");
                app.OnDestroy();
                return 1;
            }

            for (uint32_t frame = 0; frame < 100000; frame++) {
                app.OnRender();
                uint32_t current = app.GetFrameCount();
                if (frame % 200 == 0) {
                    fprintf(stderr, "[%u/%u] ", current, target);
                    fflush(stderr);
                }
                if (current >= target) {
                    fprintf(stderr, "\n");
                    break;
                }
            }

            fprintf(stderr, "[headless] Saving EXR...\n");
            app.SaveSnapshotEXR();
            if (denoise) {
                fprintf(stderr, "[headless] Denoising (OIDN)...\n");
                app.DenoiseAndSaveEXR();
            }
            app.OnDestroy();
            fprintf(stderr, "[headless] Done.\n");
            return 0;
        } else {
            return Win32Application::Run(&app, hInstance, SW_SHOW);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "[FATAL] %s\n", e.what());
        return 1;
    }
}
