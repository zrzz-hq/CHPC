#pragma once

#include "Spinnaker.h"
#include <string>
#include <vector>
#include <chrono>

using namespace Spinnaker;

class FLIRCamera 
{
public:
  struct Config
  {
    GenApi::CIntegerPtr width;
    GenApi::CIntegerPtr height;

    GenApi::CEnumerationPtr exposureMode;
    GenApi::CFloatPtr exposureTime;

    GenApi::CEnumerationPtr gainMode;
    GenApi::CFloatPtr gain;

    GenApi::CFloatPtr frameRate;
    GenApi::CBooleanPtr acquisitionFrameRateEnable;

    GenApi::CEnumerationPtr triggerMode;
    GenApi::CEnumerationPtr triggerSource;

    GenApi::CEnumerationPtr pixelFormat;
  };

  std::shared_ptr<Config> open(const std::string& Id);

  std::vector<std::string> enumCamera();
  bool start();
  void stop();
  void close();
  ImagePtr read(std::chrono::milliseconds timeout = std::chrono::milliseconds(0));
  std::shared_ptr<Config> getConfig() {return config;}
  
  FLIRCamera();
  ~FLIRCamera();

private:
  SystemPtr  mSystem;
  CameraPtr  mCam;
  CameraList mCamList;
  void startStreaming();

  int mWidth;
  int mHeight;
  int mFPS;

  std::vector<void*> buffers;
  std::shared_ptr<Config> config;
};