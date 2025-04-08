#include "FLIRCamera.h"
#include "GPU.h"
#include <memory>
#include <chrono>
#include <queue>
#include <fstream>
#include <algorithm>
#include <cctype>

#include <pthread.h>

#include <opencv2/opencv.hpp>

#include <GLFW/glfw3.h>
#include <boost/program_options.hpp>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "ui.h"


#define WIDTH 720
#define HEIGHT 540
#define FRAMERATE 100.0
#define EXPOSURE 12
#define GAIN 0.0
#define TRIGGERLINE 2


std::deque<Spinnaker::ImagePtr> imageQueue1;
pthread_mutex_t imageQueue1Mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t imageQueue1Cond = PTHREAD_COND_INITIALIZER;


std::deque<std::tuple<Spinnaker::ImagePtr, std::shared_ptr<uint8_t>, std::shared_ptr<float>>> imageQueue2;
pthread_mutex_t imageQueue2Mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t imageQueue2Cond = PTHREAD_COND_INITIALIZER;

GLuint frameTexture;
GLuint phaseTexture;
std::atomic<GPU::PhaseAlgorithm> algorithm = GPU::PhaseAlgorithm::NOVAK;
int width = WIDTH;
int height = HEIGHT;
int exposure = EXPOSURE;
int frameRate = FRAMERATE;
int triggerLine = TRIGGERLINE;
double gain = GAIN;


void cameraThreadCleanUp(void* arg)
{
    FLIRCamera* cam = reinterpret_cast<FLIRCamera*>(arg);

    cam->stop();

    std::cout << "camera thread exited" << std::endl;
}

void* cameraThreadFunc(void* arg)
{
    pthread_cleanup_push(cameraThreadCleanUp, arg);
    FLIRCamera* cam = reinterpret_cast<FLIRCamera*>(arg);

    cam->start();
    while(1)
    {
        Spinnaker::ImagePtr imagePtr = cam->read();

        pthread_mutex_lock(&imageQueue1Mutex);

        imageQueue1.push_back(imagePtr);

        pthread_cond_signal(&imageQueue1Cond);
        pthread_mutex_unlock(&imageQueue1Mutex);

        pthread_testcancel();
        
    }

    cam->stop();

    pthread_cleanup_pop(1);
    return 0;
}


void writeMatToCSV(const cv::Mat mat, const std::string& fileName){
    std::ofstream file(fileName);

    if (!file.is_open()){
        std::cout << "Error: Could not write phase map to csv file\n";
        return;
    }

    for (int i = 0; i < mat.rows; i++){
        for (int j = 0; j < mat.cols; j++){
            file << mat.at<float>(i, j);
            if (j < mat.cols-1)
                file << ',';
        }
        file << '\n';
    }
    file.close();
    std::cout << "Phase map saved to " << fileName << std::endl; 

}

void gpuThreadCleanUp(void* arg)
{
    std::cout << "gpu thread exited" << std::endl;
}

void* gpuThreadFunc(void* arg)
{
    pthread_cleanup_push(gpuThreadCleanUp, NULL);

    GPU* gpu = reinterpret_cast<GPU*>(arg);

    gpu->getCudaVersion();
    auto last = std::chrono::system_clock::now();
    while(1)
    {
        Spinnaker::ImagePtr imagePtr;
        pthread_mutex_lock(&imageQueue1Mutex);

        while(imageQueue1.size() == 0)
            pthread_cond_wait(&imageQueue1Cond, &imageQueue1Mutex);

        imagePtr = imageQueue1.back();
        imageQueue1.clear();

        pthread_mutex_unlock(&imageQueue1Mutex);

        std::pair<std::shared_ptr<uint8_t>, std::shared_ptr<float>> pair = gpu->run(imagePtr, algorithm);
        
        pthread_mutex_lock(&imageQueue2Mutex);

        imageQueue2.emplace_back(imagePtr, pair.first, pair.second);

        pthread_cond_signal(&imageQueue2Cond);
        pthread_mutex_unlock(&imageQueue2Mutex);

        pthread_testcancel();

        auto now = std::chrono::system_clock::now();
        int duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
        // std::cout << "Frame rate: " << 1000.0 / duration << std::endl;
        last = now;
    }
    pthread_cleanup_pop(1);
    return 0;
}

void renderUI()
{
    ImVec2 avail  = ImGui::GetContentRegionAvail();

    float leftWidth = avail.x * 0.25;
    float rightWidth = avail.x * 0.75;
    
    ImGui::BeginChild("Left Panel", ImVec2(leftWidth, avail.y),true);

    // Dropdown
    ImGui::Text("Choose Phase Algorithm");
    const char* algorithms[] = { "Novak", "Four Point", "Carre"};
    int selectedAlgorithm = static_cast<int>(algorithm.load());
    if(ImGui::Combo("##AlgorithmDropdown", &selectedAlgorithm, algorithms, IM_ARRAYSIZE(algorithms)))
    {
        algorithm = static_cast<GPU::PhaseAlgorithm>(selectedAlgorithm);
    }

    // Parameters Section
    ImGui::Separator();
    ImGui::Text("Parameters");

    int inputWidth = width;
    int inputHeight = height;
    ImGui::InputInt("Width", &inputWidth);
    ImGui::InputInt("Height", &inputHeight);

    int inputFrameRate = frameRate;
    int inputExposure = exposure;
    float inputGain = gain;
    ImGui::InputInt("Framerate", &inputFrameRate);
    ImGui::InputInt("Exposure", &inputExposure);
    ImGui::InputFloat("Gain", &inputGain);

    ImGui::Text("Input Device");
    const char* inputDevice[] = { "Camera 1" };
    int selectedInputDevice = 0;
    ImGui::Combo("##DeviceDropdown", &selectedInputDevice, inputDevice, IM_ARRAYSIZE(inputDevice));
    ImGui::Separator();
    // Trigger and Save Buttons
    bool triggerContinuous = false;
    ImGui::Checkbox("Trigger/Continuous", &triggerContinuous);

    int selectedTriggerLine = triggerLine;
    const char* lineSelect[] = { "Trigger Line 1", "Trigger Line 2", "Trigger Line 3",  "Trigger Line 4"};
    ImGui::Combo("##TriggerDropdown", &selectedTriggerLine, lineSelect, IM_ARRAYSIZE(lineSelect));

    ImGui::Separator();
    if (ImGui::Button("Save Phase Maps")) 
    {
        // Handle button click
    }

    ImGui::SameLine();

    ImGui::LabelText("Saved Images", "Saved Images %d", 1);

    ImGui::Separator();
    
    // Start & Stop Buttons
    if (ImGui::Button("Start", ImVec2(80, 40))) 
    {
        // Handle start
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Stop", ImVec2(80, 40))) 
    {
        // Handle stop
    }

    ImGui::EndChild();

    ImGui::SameLine();
    // ImGui::Spacing();
    ImGui::BeginChild("Right Panel", ImVec2(rightWidth, avail.y), true, ImGuiWindowFlags_HorizontalScrollbar);

    ImGui::Image((ImTextureID)frameTexture, ImVec2(width, height));
    ImGui::SameLine();
    ImGui::Image((ImTextureID)phaseTexture, ImVec2(width, height));

    ImGui::EndChild();
}

namespace po = boost::program_options;

int main(int argc, char* argv[])
{
    // po::options_description desc("CHPC options");
    // desc.add_options()
    //     ("fps", po::value<int>()->default_value(FRAMERATE), "Set the frame rate")
    //     ("width", po::value<int>()->default_value(WIDTH), "Set the frame width")
    //     ("height", po::value<int>()->default_value(HEIGHT), "Set the frame height")
    //     ("exposure", po::value<int>()->default_value(EXPOSURE), "Set the exposure time(us)")
    //     ("gain", po::value<double>()->default_value(GAIN), "Set the gain")
    //     ("algorithm", po::value<std::string>()->default_value("Novak", "Set the algorithm to create"))
    // ;

    // po::variables_map vm;
    // po::store(po::parse_command_line(argc, argv, desc), vm);
    // po::notify(vm);

    // int width = vm["width"].as<int>();
    // int height = vm["height"].as<int>();
    // int frameRate = vm["fps"].as<int>();
    int triggerLine = -1;
    // int exposure = vm["exposure"].as<int>();
    // double gain = vm["gain"].as<double>();
    // std::string algorithmStr = vm["algorithm"].as<std::string>();
    // std::transform(algorithmStr.begin(),algorithmStr.end(), algorithmStr.begin(), [](char c){return std::toupper(c);});
    // GPU::PhaseAlgorithm algorithm;
    // if(algorithmStr == "NOVAK")
    //     algorithm = GPU::PhaseAlgorithm::NOVAK;
    // else if(algorithmStr == "FOURPOINT")
    //     algorithm = GPU::PhaseAlgorithm::FOURPOINT;
    // else if(algorithmStr == "CARRE")
    //     algorithm = GPU::PhaseAlgorithm::CARRE;
    // else
    // {
    //     std::cout << "Invaild algorithm: " << algorithmStr << std::endl;
    //     return -1; 
    // }
        
    // switch(argc)
    // {
    //     case 5:
    //     triggerLine = std::stoi(argv[4]);
    //     case 4:
    //     frameRate = std::stoi(argv[3]);
    //     case 3:
    //     height = std::stoi(argv[2]);
    //     case 2:
    //     width = std::stoi(argv[1]);
    //     break;
    //     default:
    //     break;
    // }

    if(!glfwInit())
    {
        std::cout << "Failed to init glfw library" << std::endl;
        return -1;
    }

    GPU gpu(width,height,50);
    FLIRCamera cam;
    cam.open(0);
    cam.setResolution(width,height);
    cam.setFPS(frameRate);
    // cam.setExposureTime(exposure);
    cam.setPixelFormat("Mono16");
    cam.setGain(gain);

    if(triggerLine != -1)
    {
        cam.enableTrigger(TriggerSource_Line3);
    }
    else
    {
        cam.disableTrigger();
    }

    pthread_t gpuThread;
    if(pthread_create(&gpuThread, NULL, gpuThreadFunc, &gpu) == -1)
    {
        std::cout << "Failed to create GPU thread" << std::endl;
        return 1;
    }

    pthread_t cameraThread;
    if(pthread_create(&cameraThread, NULL, cameraThreadFunc, &cam) == -1)
    {
        std::cout << "Failed to create camera thread" << std::endl;
        return 1;
    }

    // glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    // int key = GLFW_KEY_UNKNOWN;
    GLFWwindow* frame = glfwCreateWindow(width * 2, height, "frame", NULL, NULL);
    // glfwSetWindowUserPointer(frame, &key);
    glfwMakeContextCurrent(frame);
    if(!frame)
    {
        std::cout << "Failed to create the main window" << std::endl;
    }

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &frameTexture);
    glBindTexture(GL_TEXTURE_2D, frameTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE16, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_SHORT , NULL);

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &phaseTexture);
    glBindTexture(GL_TEXTURE_2D, phaseTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    auto last = std::chrono::system_clock::now();
    bool spacePressed = false;
    int imageCount = 0;

    UI ui(frame, renderUI);

    while(!glfwWindowShouldClose(frame))
    {
        glfwPollEvents();
        
        std::tuple<Spinnaker::ImagePtr, std::shared_ptr<uint8_t>, std::shared_ptr<float>> tuple;

        pthread_mutex_lock(&imageQueue2Mutex);
        while(imageQueue2.size() == 0)
            pthread_cond_wait(&imageQueue2Cond, &imageQueue2Mutex);

        tuple = std::move(imageQueue2.back());
        imageQueue2.clear();

        pthread_mutex_unlock(&imageQueue2Mutex);


        glBindTexture(GL_TEXTURE_2D, frameTexture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_LUMINANCE, GL_UNSIGNED_SHORT, std::get<0>(tuple)->GetData());

        if(std::get<1>(tuple) != nullptr)
        {
            glBindTexture(GL_TEXTURE_2D, phaseTexture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, std::get<1>(tuple).get());
        }

        ui.render();

        glfwSwapBuffers(frame);
        
        int state = glfwGetKey(frame, GLFW_KEY_SPACE);
        if(state == GLFW_PRESS && !spacePressed)
        {
            spacePressed = true;
        }
        if(state == GLFW_RELEASE && spacePressed)
        {
            spacePressed = false;
            imageCount ++;
            // if(!cv::imwrite("/home/nvidia/images/" + std::to_string(imageCount) + ".png", cv::Mat(height, width, CV_32F, std::get<2>(tuple).get())))
            //     std::cout << "Failed to save image" << std::endl;
            // else
            //     std::cout << "Image saved" << std::endl;
            std::shared_ptr<float> phaseMap = std::get<2>(tuple);
            if(phaseMap != nullptr)
            {
                cv::Mat phaseMat(height, width, CV_32F);
                if(cudaMemcpy(phaseMat.data, phaseMap.get(), width*height*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                    std::cout << "Failed to copy data" << std::endl;
                else
                    writeMatToCSV(phaseMat, "/home/nvidia/images/" + std::to_string(imageCount) + ".csv");
            }
        }
        

        auto now = std::chrono::system_clock::now();
        int duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
        // std::cout << "Frame rate: " << 1000.0 / duration << std::endl;
        last = now;

    }

    pthread_cancel(gpuThread);
    pthread_cancel(cameraThread);

    pthread_join(gpuThread, NULL);
    pthread_join(cameraThread, NULL);

    imageQueue1.clear();
    imageQueue2.clear();

    cam.close();

    glfwDestroyWindow(frame);
    glfwTerminate();

    return 0;
}