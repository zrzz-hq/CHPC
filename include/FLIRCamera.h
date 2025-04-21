#pragma once

#include "Spinnaker.h"
#include <iostream>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <algorithm>

#include "buffer.h"

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

  void getVersion();

  std::shared_ptr<Config> open(size_t index);

  std::vector<std::string> enumCamera();
  bool start();
  void stop();
  void close();
  ImagePtr read();
  

  // bool setFPS(double fps);
  // bool setResolution(int width, int height);
  // bool setExposureTime(int timeNS);
  // bool enableTrigger(Spinnaker::TriggerSourceEnums line);
  // void disableTrigger();
  // bool FLIRCamera::setPixelFormat(const std::string& format);

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
};