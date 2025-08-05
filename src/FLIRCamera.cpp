#include "FLIRCamera.h"

#include <boost/log/trivial.hpp>

using namespace Spinnaker;
using namespace GenApi;

FLIRCamera::FLIRCamera()
{
    mWidth = 0;
    mHeight = 0;
    mFPS = 0;
    mSystem = System::GetInstance();

    LibraryVersion version = mSystem->GetLibraryVersion();
    BOOST_LOG_TRIVIAL(info) << "Spinnaker library version: " << version.major << "." << version.minor;
}

FLIRCamera::~FLIRCamera()
{
    stop();
    close();

    mCam = nullptr;
    mCamList.Clear();
    mSystem->ReleaseInstance();
}

std::vector<std::string> FLIRCamera::enumCamera()
{
    std::vector<std::string> cameraIds;
    mCamList = mSystem->GetCameras();
    for(size_t i=0;i<mCamList.GetSize();i++)
    {
        INodeMap& map = mCamList[i]->GetTLDeviceNodeMap();

        CStringPtr ptrModelName = map.GetNode("DeviceSerialNumber");
        if(IsAvailable(ptrModelName) && IsReadable(ptrModelName))
        {
            cameraIds.push_back(ptrModelName->GetValue().c_str());
            BOOST_LOG_TRIVIAL(info) << "Found FLIR camera: " << ptrModelName->GetValue();
        }
    }

    return cameraIds;
}

std::shared_ptr<FLIRCamera::Config> FLIRCamera::open(const std::string& Id)
{
    config = std::make_shared<Config>();

    try
    {
        mCam = mCamList.GetBySerial(Id);
    }
    catch(const Spinnaker::Exception& e)
    {
        return nullptr;
    }

    mCam->Init();

    INodeMap& nodeMap = mCam->GetNodeMap();
    INodeMap& nodeMapTLDevice = mCam->GetTLDeviceNodeMap();

    config->width = nodeMap.GetNode("Width");
    if(IsAvailable(config->width))
    {
        mWidth = config->width->GetMax();
    }

    config->height= nodeMap.GetNode("Height");
    if (IsAvailable(config->height))
    {
        mHeight =  config->height->GetMax();
    }

    config->acquisitionFrameRateEnable = nodeMap.GetNode("AcquisitionFrameRateEnable");
    if (IsAvailable(config->acquisitionFrameRateEnable) && IsWritable(config->acquisitionFrameRateEnable))
    {
        config->acquisitionFrameRateEnable->SetValue(true);
    }

    config->frameRate = nodeMap.GetNode("AcquisitionFrameRate");
    config->exposureMode = nodeMap.GetNode("ExposureAuto");
    config->exposureTime = nodeMap.GetNode("ExposureTime");
    config->gainMode = nodeMap.GetNode("GainAuto");
    config->gain = nodeMap.GetNode("Gain");
    config->pixelFormat = nodeMap.GetNode("PixelFormat");
    config->triggerMode = nodeMap.GetNode("TriggerMode");
    config->triggerSource = nodeMap.GetNode("TriggerSource");

    mCam -> TLStream.StreamBufferCountMode.SetValue(Spinnaker::StreamBufferCountModeEnum::StreamBufferCountMode_Auto);
    mCam -> TLStream.StreamBufferHandlingMode.SetValue(Spinnaker::StreamBufferHandlingModeEnum::StreamBufferHandlingMode_NewestOnly);

    return config;
}


void FLIRCamera::close()
{
    config.reset();
    
    if(mCam)
    mCam->DeInit();
}

bool FLIRCamera::start()
{
    if(mCam && mCam -> IsStreaming())
        return true;

    mCam->BeginAcquisition();

    return true;
}

void FLIRCamera::stop()
{
    if(mCam && mCam->IsStreaming())
    {
        mCam->EndAcquisition();
    }
}

ImagePtr FLIRCamera::read(std::chrono::milliseconds timeout)
{
    ImagePtr pResultImage = nullptr;

    try
    {
        pResultImage = mCam->GetNextImage(timeout.count());
    }
    catch (Spinnaker::Exception& e)
    {
        // std::cout << "Get next image timeout: " + std::string(e.what()) << std::endl;
    }

    return std::move(pResultImage);

}