#include "FLIRCamera.h"
#include "GPU.h"
#include "ui.h"

#include <memory>
#include <chrono>

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include <boost/thread.hpp>

#include <GL/gl.h>
#include <GL/glx.h>
#include <GLFW/glfw3.h>

#include <opencv2/opencv.hpp>

#define WIDTH 720
#define HEIGHT 540
#define FRAMERATE 60
#define EXPOSURETIME -1
#define GAIN 0

int main(int argc, char* argv[])
{
    std::shared_ptr<FLIRCamera> cam = std::make_shared<FLIRCamera>();
    //Start up Window

    std::shared_ptr<FLIRCamera::Config> cameraConfig;
    std::vector<std::string> deviceIds = cam->enumCamera();
    if(deviceIds.size() == 0)
    {
        ErrorWindow errorWin;
        errorWin.spin();
        return -1;
    }
    else
    {
        cameraConfig = cam->open(0);
        StartupWindow startupWin(cameraConfig);
        if(startupWin.spin() == -1)
        {
            cam->close();
            return -1;
        }
    }

    MainWindow mainWindow(cam);

    mainWindow.spin();

    return 0;
}
