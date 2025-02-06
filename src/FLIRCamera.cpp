#include "FLIRCamera.h"

using namespace Spinnaker;

FLIRCamera::FLIRCamera(){
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
}

FLIRCamera::~FLIRCamera(){
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

bool FLIRCamera::open(uint32_t devID){

 if(mCam == nullptr)
        return false;

    // Initialize camera
    mCam->Init();

    using namespace GenApi;
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

    std::cout << "Width: " << mWidth << " Height: " << mHeight << std::endl; 

    CBooleanPtr ptrAcquisitionFrameRateEnable = nodeMap.GetNode("AcquisitionFrameRateEnable");
    if (IsAvailable(ptrAcquisitionFrameRateEnable) && IsWritable(ptrAcquisitionFrameRateEnable))
    {
        ptrAcquisitionFrameRateEnable->SetValue(true);
    }

    CFloatPtr ptrFloat = nodeMap.GetNode("AcquisitionFrameRate");
    if(IsAvailable(ptrInt))
    {
        mFPS =  ptrFloat->GetValue();
        std::cout << "FPS: " << mFPS << std::endl;
    }

    // if(!mInputBuffer.allocate(mWidth, mHeight, mSurfaceFormat))
    //     return false;
}

void FLIRCamera::close()
{
    mCam->DeInit();
}

bool FLIRCamera::start()
{
    mCam->BeginAcquisition();
    // image = cv::Mat(mHeight, mWidth, CV_8UC1);
}

bool FLIRCamera::stop()
{
    mCam->EndAcquisition();
    // image.release();
}

ImagePtr FLIRCamera::read()
{
    ImagePtr pResultImage = nullptr;

    try
    {
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

    return pResultImage;

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
bool FLIRCamera::setResolution(int width, int height)
{
    using namespace Spinnaker;
    using namespace GenApi;

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
        std::cout << "Width: " << mWidth << "Height: " << mHeight << std::endl;
        return true;
    }
    catch (Spinnaker::Exception& e)
    {
        std::cerr << "Error setting resolution: " << e.what() << std::endl;
        return false;
    }

}
