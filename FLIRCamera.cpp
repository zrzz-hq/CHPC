#include "FLIRCamera.h"

using namespace Spinnaker;

FLIRCamera::FLIRCamera(){}

FLIRCamera::~FLIRCamera(){}

void FLIRCamera::getVersion(){
SystemPtr system = System::GetInstance();
LibraryVersion version = system->GetLibraryVersion();

std::cout << version.major << std::endl;


}

