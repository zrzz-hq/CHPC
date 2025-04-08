#pragma once
#include <GLFW/glfw3.h>

#include <functional>
#include <string>

class UI
{
    public:

    UI(GLFWwindow* window, std::function<void()> renderer);
    ~UI();

    void render();

    private:
    std::function<void()> renderer;
};