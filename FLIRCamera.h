#include "Spinnaker.h"
#include <iostream>

using namespace Spinnaker;
class FLIRCamera {
public:
void getVersion();
    virtual bool open(uint32_t devID);
    virtual bool start();
    virtual bool stop();
    virtual void close();

FLIRCamera();
~FLIRCamera();

private:
  SystemPtr  mSystem;
    CameraList mCamList;
    CameraPtr  mCam     = nullptr;
    void startStreaming();
};
