#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

class DXRApp;

class Win32Application
{
public:
    static int Run(DXRApp *pApp, HINSTANCE hInstance, int nCmdShow);
    static HWND GetHwnd() { return s_hwnd; }

private:
    static LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
    static HWND s_hwnd;
};