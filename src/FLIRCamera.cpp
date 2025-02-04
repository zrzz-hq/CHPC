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
    nodeMap = mCam->GetNodeMap();
    nodeMapTLDevice = mCam->GetTLDeviceNodeMap();

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
    image = cv::Mat(mHeight, mWidth, CV_8UC1);
}

bool FLIRCamera::stop()
{
    mCam->EndAcquisition();
    image.release();
}

cv::Mat& FLIRCamera::read()
{
    try
    {
        // Retrieve next received image
        ImagePtr pResultImage = mCam->GetNextImage(1000);

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
            // memcpy(image.data, src, pResultImage->GetBufferSize());
            cv::imshow("frame", cv::Mat(mHeight, mWidth, CV_8UC1, src));
            
            size_t sz = pResultImage->GetImageSize();
        }

        // Release image
        pResultImage->Release();

    }
    catch (Spinnaker::Exception& e)
    {
        std::cout << "Error: " << e.what();
    }

    return image;

}

    bool FLIRCamera::setFPS(int targetFPS){
        using namespace Spinnaker;
        using namespace GenApi;
        if (mCam == nullptr)
            return false;
        
        CIntegerPtr ptrWidth = nodeMap.GetNode("Width");
        if (IsAvailable){

        }

    }
    bool setResolution(int width, int height){


    }
