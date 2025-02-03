#include "FLIRCamera.h"

using namespace Spinnaker;

FLIRCamera::FLIRCamera(){
     try
    {
        mSystem = System::GetInstance();
        mCamList = mSystem->GetCameras();
        const unsigned int numCameras = mCamList.GetSize();
        if (numCameras == 0)
        {
            // Clear camera list before releasing system
            mCamList.Clear();

            // Release system
            mSystem->ReleaseInstance();
            return;
        }
        mCam = mCamList.GetByIndex(0);
        std::cout << "numCameras: " << numCameras << std::endl;
    }
    catch (Spinnaker::Exception& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
        return;
    }
}

FLIRCamera::~FLIRCamera(){}

void FLIRCamera::getVersion(){
SystemPtr system = System::GetInstance();
LibraryVersion version = system->GetLibraryVersion();

std::cout << version.major << std::endl;


}

