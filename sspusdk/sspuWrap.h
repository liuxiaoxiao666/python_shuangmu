#pragma once
#define DLL_API _declspec(dllexport)

DLL_API int sspu_start();

DLL_API void sspu_stop();

DLL_API int add(int x, int y);