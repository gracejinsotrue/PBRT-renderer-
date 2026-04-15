#include "Win32Application.h"
#include "DXRApp.h"

HWND Win32Application::s_hwnd = nullptr;

LRESULT CALLBACK Win32Application::WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    DXRApp *pApp = reinterpret_cast<DXRApp *>(GetWindowLongPtr(hWnd, GWLP_USERDATA));

    switch (message)
    {
    case WM_CREATE:
    {
        auto *pCreateStruct = reinterpret_cast<LPCREATESTRUCT>(lParam);
        SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pCreateStruct->lpCreateParams));
        return 0;
    }
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE)
        {
            DestroyWindow(hWnd);
        }
        else if (pApp)
        {
            pApp->OnKeyDown(static_cast<UINT8>(wParam));
        }
        return 0;
    case WM_KEYUP:
        if (pApp)
            pApp->OnKeyUp(static_cast<UINT8>(wParam));
        return 0;
    case WM_RBUTTONDOWN:
        if (pApp)
        {
            SetCapture(hWnd);
            pApp->OnMouseDown(1, (int)(short)LOWORD(lParam), (int)(short)HIWORD(lParam));
        }
        return 0;
    case WM_RBUTTONUP:
        if (pApp)
        {
            ReleaseCapture();
            pApp->OnMouseUp(1, (int)(short)LOWORD(lParam), (int)(short)HIWORD(lParam));
        }
        return 0;
    case WM_MOUSEMOVE:
        if (pApp)
            pApp->OnMouseMove((int)(short)LOWORD(lParam), (int)(short)HIWORD(lParam));
        return 0;
    case WM_PAINT:
        // Handled in idle loop; just validate
        ValidateRect(hWnd, nullptr);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }

    return DefWindowProc(hWnd, message, wParam, lParam);
}

int Win32Application::Run(DXRApp *pApp, HINSTANCE hInstance, int nCmdShow)
{
    WNDCLASSEX wc = {};
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = L"NoriDXRClass";
    RegisterClassEx(&wc);

    RECT rc = {0, 0, static_cast<LONG>(pApp->GetWidth()), static_cast<LONG>(pApp->GetHeight())};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

    s_hwnd = CreateWindowEx(
        0, L"NoriDXRClass", pApp->GetTitle(),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left, rc.bottom - rc.top,
        nullptr, nullptr, hInstance, pApp);

    pApp->OnInit();
    ShowWindow(s_hwnd, nCmdShow);

    MSG msg = {};
    while (msg.message != WM_QUIT)
    {
        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else
        {
            // do continous render loop
            if (pApp)
            {
                try
                {
                    pApp->OnUpdate();
                    pApp->OnRender();
                }
                catch (const std::exception &e)
                {
                    fprintf(stderr, "[RENDER ERROR] %s\n", e.what());
                    DestroyWindow(s_hwnd);
                }
            }
        }
    }
    pApp->OnDestroy();
    return static_cast<int>(msg.wParam);
}