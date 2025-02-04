#include "FLIRCamera.h"
#include "GPU.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

int main(){

    std::unique_ptr<FLIRCamera> cam(new FLIRCamera());
    cam->open(0);

    // std::unique_ptr<GPU> gpu(new GPU());
    // gpu->getCudaVersion();
    
    std::cout << "Opencv version " << cv::getVersionMajor() << std::endl;
    
    cam->start();
    while(1)
    {
        // cv::Mat& image = cam->read();
        // cv::imshow("image", image);
        cam->read();
        if(cv::waitKey(1) == 'q')
            break;
    }

    cv::destroyAllWindows();
    
    cam->stop();

    cam->close();
    return 0;



}
