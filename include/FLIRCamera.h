#include "Spinnaker.h"
#include <iostream>

using namespace Spinnaker;
class FLIRCamera {
public:
void getVersion();
    bool open(uint32_t devID);
    bool start();
    bool stop();
    void close();

FLIRCamera();
~FLIRCamera();

private:
  SystemPtr  mSystem;
    CameraList mCamList;
    CameraPtr  mCam     = nullptr;
    void startStreaming();

    int mWidth;
    int mHeight;
};
