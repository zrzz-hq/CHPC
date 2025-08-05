#include "FLIRCamera.h"
#include "GPU.h"
#include "ui.h"

#include <memory>
#include <chrono>

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include <boost/thread.hpp>

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

    cameraConfig = cam->open(deviceIds[0]);
    while(1)
    {
        {
            StartupWindow startupWin(cameraConfig);
            if(startupWin.spin() == -1)
            {
                break;
            }
        }

        {
            MainWindow mainWindow(cameraConfig->width->GetValue(), cameraConfig->height->GetValue());

            boost::thread gpuThread([cam, &mainWindow]()
            {
                cam->start();
                try
                {
                    while(1)
                    {
                        Spinnaker::ImagePtr image = cam->read(std::chrono::milliseconds(100));
                        mainWindow.processImage(image);
                        boost::this_thread::interruption_point();
                    }
                }
                catch(const boost::thread_interrupted& i)
                {
                    
                }

                cam->stop();
            });

            mainWindow.spin();

            gpuThread.interrupt();
            gpuThread.join();
        }
    }

    cam->close();
    return 0;
}
