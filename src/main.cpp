#include "FLIRCamera.h"
#include "GPU.h"
#include <unique_ptr>

int main(){

std::unique_ptr<FLIRCamera> cam = std::make_unique<FLIRCamera>();
//cam->getVersion();
cam->open(0);

GPU* gpu = new GPU();
//gpu->getCudaVersion();

delete cam;
delete gpu;
return 0;


}
