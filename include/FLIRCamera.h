#include "Spinnaker.h"
#include <iostream>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <algorithm>

using namespace Spinnaker;
using namespace GenApi;

class FLIRCamera 
{
public:
  void getVersion();
  bool open(uint32_t devID);
  bool start();
  void stop();
  void close();
  ImagePtr read();
  bool setExposureTime(int timeNS);
  bool setPixelFormat(const std::string& format);
  bool setFPS(double fps);
  bool setGain(double value);
  bool setResolution(int width, int height);
  bool enableTrigger(Spinnaker::TriggerSourceEnums line);
  void disableTrigger();

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