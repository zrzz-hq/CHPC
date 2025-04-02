#include "FLIRCamera.h"
#include "cuda_runtime.h"

FLIRCamera::FLIRCamera()
{
    try
    {
        mWidth = 0;
        mHeight = 0;
        mFPS = 0;
        mSystem = System::GetInstance();
        mCamList = mSystem->GetCameras();
        const unsigned int numCameras = mCamList.GetSize();
        if (numCameras == 0)
        {
            // Clear camera list before releasing system
            mCamList.Clear();

            // Release system
            mSystem->ReleaseInstance();
            throw std::runtime_error("No camera dectected!\n");
           
        }
        mCam = mCamList.GetByIndex(0);
       // std::cout << "numCameras: " << numCameras << std::endl;
    }
    catch (Spinnaker::Exception& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
        return;
    }

    // mCam->TriggerSource.SetValue(Spinnaker::TriggerSourceEnums::TriggerSource_Software);
    mCam -> TLStream.StreamBufferCountMode.SetValue(Spinnaker::StreamBufferCountModeEnum::StreamBufferCountMode_Auto);
    mCam -> TLStream.StreamBufferHandlingMode.SetValue(Spinnaker::StreamBufferHandlingModeEnum::StreamBufferHandlingMode_OldestFirstOverwrite);
}

FLIRCamera::~FLIRCamera(){
    stop();
    close();

    mCam = nullptr;
    // Clear camera list before releasing system
    mCamList.Clear();
    // Release system
    mSystem->ReleaseInstance();
}

void FLIRCamera::getVersion(){
SystemPtr system = System::GetInstance();
LibraryVersion version = system->GetLibraryVersion();

std::cout << version.major << std::endl;
}

bool FLIRCamera::open(uint32_t devID)
{

 if(mCam == nullptr)
        return false;

    // Initialize camera
    mCam->Init();

    INodeMap& nodeMap = mCam->GetNodeMap();
    INodeMap& nodeMapTLDevice = mCam->GetTLDeviceNodeMap();

      CIntegerPtr ptrInt = nodeMap.GetNode("Width");
    if(IsAvailable(ptrInt))
    {
        mWidth = ptrInt->GetMax();
    }

    ptrInt = nodeMap.GetNode("Height");
    if (IsAvailable(ptrInt))
    {
        mHeight =  ptrInt->GetMax();
    }

    std::cout << "Maximum width: " << mWidth << " Maximum height: " << mHeight << std::endl; 

    CBooleanPtr ptrAcquisitionFrameRateEnable = nodeMap.GetNode("AcquisitionFrameRateEnable");
    if (IsAvailable(ptrAcquisitionFrameRateEnable) && IsWritable(ptrAcquisitionFrameRateEnable))
    {
        ptrAcquisitionFrameRateEnable->SetValue(true);
    }

    CFloatPtr framerate = nodeMap.GetNode("AcquisitionFrameRate");
    if(IsAvailable(framerate))
    {
        mFPS =  framerate->GetValue();
        std::cout << "FPS: " << mFPS << std::endl;
    }

    mCam -> TLStream.StreamBufferCountMode.SetValue(Spinnaker::StreamBufferCountModeEnum::StreamBufferCountMode_Auto);
    mCam -> SetBufferOwnership(Spinnaker::BufferOwnership::BUFFER_OWNERSHIP_USER);
    // if(!mInputBuffer.allocate(mWidth, mHeight, mSurfaceFormat))
    //     return false;
}

void FLIRCamera::close()
{
    mCam->DeInit();
}

bool FLIRCamera::start()
{
    if(mCam -> IsStreaming())
        return true;
    
    size_t bufferSize = ((mWidth * mHeight * (mCam->PixelSize.GetValue() + 8 - 1) / 8 * 8 + 1024 - 1) / 1024) * 1024;
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
    if(mCam->IsStreaming())
    {
        mCam->EndAcquisition();
    }

    for(void* buffer : buffers)
    {
        cudaFree(buffer);
    }
}

bool FLIRCamera::enableTrigger(Spinnaker::TriggerSourceEnums line)
{
    mCam->TriggerSelector.SetValue(TriggerSelector_FrameBurstStart);
    mCam->TriggerSource.SetValue(line);
    mCam->TriggerActivation.SetValue(TriggerActivation_RisingEdge);
    mCam->TriggerMode.SetValue(TriggerMode_On);
    return mCam->TriggerMode.GetValue() == TriggerMode_On;
}

void FLIRCamera::disableTrigger()
{
    mCam->TriggerMode.SetValue(TriggerMode_Off);
    std::cout << "Trigger Mode Enable: " << mCam->TriggerMode.GetValue() << std::endl;
    INodeMap& nodeMap = mCam->GetNodeMap();
    CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
    if (IsAvailable(ptrAcquisitionMode)){
        CEnumEntryPtr ptrAcquisitionContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
        ptrAcquisitionMode->SetIntValue(ptrAcquisitionContinuous->GetValue());
    }
    else{
        std::cout << "Failed to get Acquisition Mode" << std::endl;
    }
}

bool FLIRCamera::setGain(double value)
{
    INodeMap& nodeMap = mCam->GetNodeMap();

    CEnumerationPtr gainAuto = nodeMap.GetNode("GainAuto");
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

ImagePtr FLIRCamera::read()
{
    ImagePtr pResultImage = nullptr;

    try
    {
        // std::cout << "Input buffer count: " << mCam->TLStream.StreamInputBufferCount.GetValue() << std::endl;
        // std::cout << "Lost buffer count: " << mCam->TLStream.StreamLostFrameCount.GetValue() << std::endl;
        // Retrieve next received image
        pResultImage = mCam->GetNextImage();

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
        throw std::runtime_error("Spinnaker Error: " + std::string(e.what()));
    }

    return std::move(pResultImage);

}

bool FLIRCamera::setFPS(double fps){

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
bool FLIRCamera::setResolution(int width, int height)
{

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
            CEnumEntryPtr exposureAutoOff = exposureAuto->GetEntryByName("On");
            exposureAuto->SetIntValue(exposureAutoOff->GetValue());
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
