#include "GPU.h"

GPU::GPU(){};
GPU::~GPU(){};
void GPU::getCudaVersion(){
int rVersion = 0;
//int dVersion = 0;

cudaError_t runtimeStatus = cudaRuntimeGetVersion(&rVersion);
std:: cout << rVersion << std::endl;


}
