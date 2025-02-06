#include "Spinnaker.h"
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <algorithm>

using namespace Spinnaker;
class FLIRCamera {
public:
void getVersion();
    bool open(uint32_t devID);
    bool start();
    bool stop();
    void close();
    ImagePtr read();
    bool setFPS(double fps);
    bool setResolution(int width, int height);

FLIRCamera();
~FLIRCamera();

private:
  SystemPtr  mSystem;
    CameraList mCamList;
    CameraPtr  mCam     = nullptr;
    void startStreaming();

    int mWidth;
    int mHeight;
    int mFPS;

    cv::Mat image;
};
