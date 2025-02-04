#include "FLIRCamera.h"
#include "GPU.h"

int main(){

FLIRCamera* cam = new FLIRCamera();
//cam->getVersion();

GPU* gpu = new GPU();
//gpu->getCudaVersion();

delete cam;
delete gpu;
return 0;


}
