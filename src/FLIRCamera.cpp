#include "FLIRCamera.h"
#include "cuda_runtime.h"

using namespace Spinnaker;
using namespace GenApi;

FLIRCamera::FLIRCamera()
{
    try
    {
        mWidth = 0;
        mHeight = 0;
        mFPS = 0;
        mSystem = System::GetInstance();
        mCamList = mSystem->GetCameras();
            
       // std::cout << "numCameras: " << numCameras << std::endl;
    }
    catch (Spinnaker::Exception& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
        return;
    }
}

FLIRCamera::~FLIRCamera()
{
    stop();
    close();

    mCam = nullptr;
    // Clear camera list before releasing system
    mCamList.Clear();
    // Release system
    mSystem->ReleaseInstance();
}

std::vector<std::string> FLIRCamera::enumCamera()
{
    std::vector<std::string> cameraIds;
    mCamList = mSystem->GetCameras();
    for(size_t i=0;i<mCamList.GetSize();i++)
    {
        // mCamList[i]->Init();
        INodeMap& map = mCamList[i]->GetTLDeviceNodeMap();

        CStringPtr ptrModelName = map.GetNode("DeviceSerialNumber");
        if(IsAvailable(ptrModelName) && IsReadable(ptrModelName))
        {
            cameraIds.push_back(ptrModelName->GetValue().c_str());
            std::cout<< "Found camera: " << ptrModelName->GetValue() << std::endl;
        }

        // cameraIds.push_back(mCamList[i]->DeviceSerialNumber().c_str());
        // std::cout<< "Found camera: " << mCamList[i]->DeviceSerialNumber() << std::endl;
        // mCamList[i]->DeInit();
    }

    return cameraIds;
}

void FLIRCamera::getVersion()
{
SystemPtr system = System::GetInstance();
LibraryVersion version = system->GetLibraryVersion();

std::cout << version.major << std::endl;
}

std::shared_ptr<FLIRCamera::Config> FLIRCamera::open(size_t index)
{
    std::shared_ptr<Config> config = std::make_shared<Config>();
    mCam = mCamList.GetByIndex(index);

    if(!mCam)
        return nullptr;

    // Initialize camera
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

    std::cout << "Maximum width: " << mWidth << " Maximum height: " << mHeight << std::endl; 

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
    // mCam -> SetBufferOwnership(Spinnaker::BufferOwnership::BUFFER_OWNERSHIP_USER);

    return config;
}


void FLIRCamera::close()
{
    if(mCam)
    mCam->DeInit();
}

bool FLIRCamera::start()
{
    if(mCam && mCam -> IsStreaming())
    return true;

size_t bufferSize = ((mWidth * mHeight + 1024 - 1) / 1024) * 1024;
// size_t bufferSize = mWidth * mHeight;
unsigned userBufferNum = 50;
for(int i=0; i<userBufferNum; i++)
{
    void* hostBuffer;
    cudaError_t error = cudaMallocHost(&hostBuffer, bufferSize);
    if(error != cudaSuccess)
    {
        std::cout << "Failed to allocate image buffer: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    buffers.push_back(hostBuffer);
}

mCam->SetUserBuffers(buffers.data(), userBufferNum, bufferSize);
mCam->BeginAcquisition();

std::cout << "Maximum number of buffers: " << mCam -> TLStream.StreamBufferCountMax.GetValue() << std::endl;
std::cout << "Number of input buffers: " << mCam -> TLStream.StreamInputBufferCount.GetValue() << std::endl;

return true;
}

void FLIRCamera::stop()
{
    if(mCam && mCam->IsStreaming())
    {
        mCam->EndAcquisition();
        
        for(void* buffer : buffers)
        {
            cudaFree(buffer);
        }
    }
}

ImagePtr FLIRCamera::read()
{
    ImagePtr pResultImage = nullptr;

    try
    {
        // std::cout << "Input buffer count: " << mCam->TLStream.StreamInputBufferCount.GetValue() << std::endl;
        // std::cout << "Lost buffer count: " << mCam->TLStream.StreamLostFrameCount.GetValue() << std::endl;
        // Retrieve next received image
        pResultImage = mCam->GetNextImage(100);

        // Ensure image is complete
        if (pResultImage->IsIncomplete())
        {
            // Retrieve and print the image status description
            throw std::runtime_error("Image incomplete: " + 
                std::string(Image::GetImageStatusDescription(pResultImage->GetImageStatus())));
        }
        else
        {
            void* src = pResultImage->GetData();
            
            // size_t sz = pResultImage->GetImageSize();
        }

        // Release image
        // pResultImage->Release();

    }
    catch (Spinnaker::Exception& e)
    {
        // std::cout << "Get next image timeout: " + std::string(e.what()) << std::endl;
    }

    return std::move(pResultImage);

}

//Set Functions
/*
bool FLIRCamera::setGain(double value)
{
    INodeMap& nodeMap = mCam->GetNodeMap();

   
    if(IsAvailable(gainAuto) && IsWritable(gainAuto))
    {
        gainAuto->FromString("Off");
    }

    CFloatPtr gain = nodeMap.GetNode("Gain");
    if(IsAvailable(gain) && IsWritable(gain))
    {
        double minGain = gain->GetMin();
        double maxGain = gain->GetMax();

        gain->SetValue(std::min(std::max(value, minGain), maxGain));
    }

    std::cout << "Gain: " << gain->GetValue() << std::endl;
}

bool FLIRCamera::enableTrigger(Spinnaker::TriggerSourceEnums line)//Uses Spinnaker instead of GenApi
{
    mCam->TriggerSelector.SetValue(TriggerSelector_FrameBurstStart);
    mCam->TriggerSource.SetValue(line);
    mCam->TriggerActivation.SetValue(TriggerActivation_RisingEdge);
    mCam->TriggerMode.SetValue(TriggerMode_On);
    return mCam->TriggerMode.GetValue() == TriggerMode_On;
}pixelFormat = nodeMap.GetNode("PixelFormat");

void FLIRCamera::disableTrigger()//Uses Spinnaker instead of GenApi
{
    mCam->TriggerMode.SetValue(TriggerMode_Off);
}
bool FLIRCamera::setPixelFormat(const std::string& format)
{
    INodeMap& nodeMap = mCam->GetNodeMap();
    
    CEnumerationPtr pixelFormat = nodeMap.GetNode("PixelFormat");
    if(IsAvailable(pixelFormat) && IsWritable(pixelFormat))
    {
        CEnumEntryPtr pixelFormatMono16 = pixelFormat->GetEntryByName("Mono16");
        if(IsAvailable(pixelFormatMono16) && IsReadable(pixelFormatMono16))
        {
            pixelFormat->SetIntValue(pixelFormatMono16->GetValue());
        }
        else
        {
            std::cout << "The camera does not support mono16 format" << std::endl;
            return false;
        }
    }
    else
    {
        std::cout << "Failed to set the pixel format";
        return false;
    }
    
    return true;
}

bool FLIRCamera::setFPS(double fps){
    using namespace Spinnaker;
    using namespace GenApi;
    
    if (mCam == nullptr)
    return false; // Camera not available
    
    try
    {
        INodeMap& nodeMap = mCam->GetNodeMap();
        // Enable manual frame rate control
        CBooleanPtr ptrFrameRateEnable = nodeMap.GetNode("AcquisitionFrameRateEnable");
        if (IsAvailable(ptrFrameRateEnable) && IsWritable(ptrFrameRateEnable))
        {
            ptrFrameRateEnable->SetValue(true);
        }
        else
        {
            return false;
        }

        // Set the frame rate
        CFloatPtr ptrFrameRate = nodeMap.GetNode("AcquisitionFrameRate");
        if (IsAvailable(ptrFrameRate) && IsWritable(ptrFrameRate))
        {
            fps = std::min(fps, ptrFrameRate->GetMax()); // Ensure within max limit
            ptrFrameRate->SetValue(fps);
        }
        else
        {
            return false;
        }
        CFloatPtr ptrFloat = nodeMap.GetNode("AcquisitionFrameRate");
        if(IsAvailable(ptrFloat))
        {
            mFPS =  ptrFloat->GetValue();
            std::cout << "FPS: " << mFPS << std::endl;
        }
        return true;
    }
    catch (Spinnaker::Exception& e)
    {
        std::cerr << "Error setting FPS: " << e.what() << std::endl;
        return false;
    }
}

bool FLIRCamera::setResolution(int width, int height){
    
if (mCam == nullptr)
return false; // Camera not available

try
{
        INodeMap& nodeMap = mCam->GetNodeMap();
        // Set Width
        CIntegerPtr ptrWidth = nodeMap.GetNode("Width");
        if (IsAvailable(ptrWidth) && IsWritable(ptrWidth))
        {
            width = std::min(width, (int)ptrWidth->GetMax()); // Ensure within limits
            ptrWidth->SetValue(width);
        }
        else
        {
            return false;
        }
        // Set Height
        CIntegerPtr ptrHeight = nodeMap.GetNode("Height");
        if (IsAvailable(ptrHeight) && IsWritable(ptrHeight))
        {
            height = std::min(height, (int)ptrHeight->GetMax()); // Ensure within limits
            ptrHeight->SetValue(height);
        }
        else
        {
            return false;
        }

        CIntegerPtr ptrInt = nodeMap.GetNode("Width");
        if(IsAvailable(ptrInt))
        {
            mWidth = ptrInt->GetValue();
        }

        ptrInt = nodeMap.GetNode("Height");
        if (IsAvailable(ptrInt))
        {
            mHeight =  ptrInt->GetValue();
        }
        std::cout << "Width: " << mWidth << " Height: " << mHeight << std::endl;
        return true;
    }
    catch (Spinnaker::Exception& e)
    {
        std::cerr << "Error setting resolution: " << e.what() << std::endl;
        return false;
    }

}

bool FLIRCamera::setExposureTime(int timeNS)
{
    INodeMap& nodeMap = mCam->GetNodeMap();
    if(timeNS < 0)
    {
        CEnumerationPtr exposureAuto = nodeMap.GetNode("ExposureAuto");
        if(IsAvailable(exposureAuto) && IsWritable(exposureAuto))
        {
            CEnumEntryPtr exposureContinous = exposureAuto->GetEntryByName("Continuous");
            exposureAuto->SetIntValue(exposureContinous->GetValue());
        }
    }
    else
    {
        CEnumerationPtr exposureAuto = nodeMap.GetNode("ExposureAuto");
        if(IsAvailable(exposureAuto) && IsWritable(exposureAuto))
        {
            CEnumEntryPtr exposureAutoOff =exposureAuto->GetEntryByName("Off");
            exposureAuto->SetIntValue(exposureAutoOff->GetValue());
        }

        CFloatPtr exposureTime = nodeMap.GetNode("ExposureTime");
        if(IsAvailable(exposureTime) && IsWritable(exposureTime))
        {
            exposureTime->SetValue(timeNS);
            std::cout << "Exposure time: " << exposureTime->GetValue() << std::endl; 
        }
    }
}
*/
