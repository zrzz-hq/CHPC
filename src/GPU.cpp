#include "GPU.h"

GPU::GPU(int width, int height){
    eleCount = 0;
    N = width * height;
    blockPerGrid = (N + 256 - 1) / 256;
};
GPU::~GPU(){};
void GPU::getCudaVersion(){
int rVersion = 0;
//int dVersion = 0;

cudaError_t runtimeStatus = cudaRuntimeGetVersion(&rVersion);
std:: cout << rVersion << std::endl;


cudaError_t error = cudaMallocManaged(&outputBuffer, N*sizeof(float));
if(error != cudaSuccess)
{
    throw std::runtime_error("Failed to allocate cuda memory: " + std::string(cudaGetErrorString(error)));
}

}

float* GPU::runNovak(Spinnaker::ImagePtr image)
{
    
    if (eleCount < 5){
        buffer[eleCount] = image;
        eleCount++;
        return nullptr;
        }

    buffer[4]->Release();
    buffer[4] = buffer[3];
    buffer[3] = buffer[2];
    buffer[2] = buffer[1];
    buffer[1] = buffer[0];
    buffer[0] = image;


    
    compute_phase<<<blockPerGrid,256>>>(reinterpret_cast<uint8_t*>(buffer[0]->GetData()),
                                        reinterpret_cast<uint8_t*>(buffer[1]->GetData()),
                                        reinterpret_cast<uint8_t*>(buffer[2]->GetData()),
                                        reinterpret_cast<uint8_t*>(buffer[3]->GetData()),
                                        reinterpret_cast<uint8_t*>(buffer[4]->GetData()),
                                        outputBuffer,
                                        N);
    cudaDeviceSynchronize();
    return outputBuffer;
}