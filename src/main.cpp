#include "FLIRCamera.h"
#include "GPU.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <chrono>

int main(){

    std::unique_ptr<FLIRCamera> cam(new FLIRCamera());
    cam->open(0);
    cam->setResolution(800,600);
    cam->setFPS(120.0);

    std::unique_ptr<GPU> gpu(new GPU(800,600));
    gpu->getCudaVersion();
    std::cout << "Opencv version " << cv::getVersionMajor() << std::endl;
    
    cam->start();
    cv::namedWindow("frame");
    auto last = std::chrono::system_clock::now();
    while(1)
    {
       
        Spinnaker::ImagePtr imagePtr = cam->read();
        cv::Mat image(imagePtr->GetHeight(),imagePtr->GetWidth(),CV_8UC1, imagePtr->GetData());

        auto now = std::chrono::system_clock::now();
        int duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
        last = now;

        float* phase = gpu->runNovak(imagePtr);
        if(phase != nullptr)
        {
            // cv::Mat phaseImage(800,600,CV_32FC1,phase);
            // cv::imshow("phase", phaseImage);
        }

        cv::putText(image, std::to_string(1000.0/duration), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 2);
        cv::imshow("frame",image);

        if(cv::waitKey(1) == 'q')
            break;
    }

    cv::destroyAllWindows();
    
    cam->stop();

    cam->close();
    return 0;



}
