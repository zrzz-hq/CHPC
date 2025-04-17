#pragma once
#include <GLFW/glfw3.h>

#include <functional>
#include <vector>
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

struct StartupParameters
{
    int width;
    int height;
    int exposureTime;

    float gain;
    int frameRate;
    int triggerLine;

    const char** deviceNames;
    unsigned nDevices;
    int deviceIndex;
};

struct MainParameters
{
    int algorithmIndex;

    int nSavedImages;

    const char* algorithms[3] = { "Carre", "Novak", "Four Point" };
    unsigned nAlgorithms;

    std::function<void(void)> onSaveImage;

    GLuint imageTexture;
    GLuint phaseTexture;
};

void startupUI(StartupParameters& params);
void errorUI(bool& shouldClose);
void mainUI(MainParameters& params);

