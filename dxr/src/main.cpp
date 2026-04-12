#include "Win32Application.h"
#include "DXRApp.h"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: nori-dxr <scene.xml>\n");
        fprintf(stderr, "Example: nori-dxr ..\\scenes\\a4\\cbox\\cbox_mis.xml\n");
        return 1;
    }

    try {
        HINSTANCE hInstance = GetModuleHandle(nullptr);
        DXRApp app(argv[1]);
        return Win32Application::Run(&app, hInstance, SW_SHOW);
    } catch (const std::exception& e) {
        fprintf(stderr, "[FATAL] %s\n", e.what());
        return 1;
    }
}
