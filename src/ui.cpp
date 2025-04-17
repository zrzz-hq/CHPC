#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "ui.h"

#include <cstring>
#include <algorithm>
#include <fstream>

using namespace Spinnaker;
using namespace GenApi;

#define IGNORE_SPINNAKER_ERROR(expr)           \
    do {                                       \
        try {                                  \
            expr;                              \
        } catch (const Spinnaker::Exception& e) { \
            std::cerr << "[Spinnaker Error] "  \
                      << e.GetErrorMessage()   \
                      << std::endl;            \
        }                                      \
    } while (0)
    
WindowBase::WindowBase(size_t width, size_t height, const std::string& name)
{
    if(windowCount++ == 0)
    {
        glfwInit();
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    //io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    (void)io;

    ImGui::StyleColorsLight();  // Use Light Mode Theme

    // Setup platform/renderer bindings
    frame = glfwCreateWindow(width, height, name.c_str(), NULL, NULL);
    glfwMakeContextCurrent(frame);
    ImGui_ImplGlfw_InitForOpenGL(frame, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

WindowBase::~WindowBase()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(frame);

    if(--windowCount == 0)
    {
        glfwTerminate();
    }
}

size_t WindowBase::windowCount = 0;

void WindowBase::spinOnce()
{
    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);

    render();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(frame);
}

int WindowBase::spin()
{
    glfwMakeContextCurrent(frame);

    while(!glfwWindowShouldClose(frame) && !shouldClose)
    {
        spinOnce();
    }

    return ret;
}

bool WindowBase::ok()
{
    return !glfwWindowShouldClose(frame) && !shouldClose;
}

void WindowBase::exit(int returnValue)
{
    shouldClose = true;
    ret = returnValue;
}

ErrorWindow::ErrorWindow():
    WindowBase(200, 150, "Error")
{
}

ErrorWindow::~ErrorWindow()
{

}

void ErrorWindow::render(){
    ImGui::Begin("Error", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse);
    ImGui::SetWindowFontScale(1.5f); 
    //ImVec2 windowSize = ImGui::GetWindowSize();
    const char* message = "No Camera Detected!";
    //float widgetWidth = ImGui::CalcTextSize(message).x;
    
    //ImGui::SetCursorPosX((windowSize.x - widgetWidth) * 0.5f);
    ImGui::Text("%s", message);
    ImGui::Dummy(ImVec2(0.0f, 20.0f));
    if (ImGui::Button("Close Window"))
    {
        exit(0);
    }
    ImGui::End();
}

void WindowBase::render()
{
    ImGui::Begin("Main Panel", nullptr, ImGuiWindowFlags_AlwaysAutoResize
    | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse
    | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus );

    ImGui::Text("Empty");

    ImGui::End();
}

StartupWindow::StartupWindow(std::shared_ptr<FLIRCamera::Config> config):
    WindowBase(300, 250, "Options"),
    config_(config)
{
    getEnumerate(exposureModeEnum, config_->exposureMode);
    getEnumerate(triggerModeEnum, config_->triggerMode);
    getEnumerate(triggerSourceEnum, config_->triggerSource);
    getEnumerate(gainModeEnum, config_->gainMode);
}

void StartupWindow::getEnumerate(std::pair<char**, int>& penum, GenApi::CEnumerationPtr cenum)
{
    NodeList_t entries;
    cenum->GetEntries(entries);
    penum.first = new char*[entries.size()];
    penum.second = entries.size();

    for(size_t i=0;i<penum.second;i++)
    {
        auto entryName = static_cast<CEnumEntryPtr>(entries[i])->GetSymbolic();
        penum.first[i] = new char[entryName.size() + 1];
        std::strcpy(penum.first[i], entryName.c_str());
    }

}

void StartupWindow::destroyEnumerate(std::pair<char**, int>& penum)
{
    for(size_t i=0;i<penum.second;i++)
    {
        delete[] penum.first[i];
    }
    delete[] penum.first;
    penum.second = 0;
}

StartupWindow::~StartupWindow()
{
    destroyEnumerate(exposureModeEnum);
    destroyEnumerate(triggerModeEnum);
    destroyEnumerate(triggerSourceEnum);
    destroyEnumerate(gainModeEnum);
}

void StartupWindow::render()
{
    ImGui::Begin("Main Panel", nullptr, ImGuiWindowFlags_AlwaysAutoResize
    | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse
    | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus ); 

    int width = config_->width->GetValue();
    int height = config_->height->GetValue();
    ImGui::Text("Width"); ImGui::SameLine(120);
    if(ImGui::InputInt("##Width", &width, 4))
    {
        IGNORE_SPINNAKER_ERROR(config_->width->SetValue(width));
    }
    ImGui::Text("Height"); ImGui::SameLine(120);
    if(ImGui::InputInt("##Height", &height))
    {
        IGNORE_SPINNAKER_ERROR(config_->height->SetValue(height));
    }

    ImGui::Separator();

    float frameRate = config_->frameRate->GetValue();
    ImGui::Text("Framerate");
    ImGui::SameLine(120);
    if(ImGui::InputFloat("##Framerate", &frameRate))
    {
        config_->acquisitionFrameRateEnable->SetValue(true);//Might need to move this to trigger mode(continuous)
        IGNORE_SPINNAKER_ERROR(config_->frameRate->SetValue(frameRate));
    }

    ImGui::Separator();

    float exposureTime = config_->exposureTime->GetValue();
    int exposureModeIndex = config_->exposureMode->GetCurrentEntry()->GetValue();
    ImGui::Text("Exposure Mode"); ImGui::SameLine(120);
    if(ImGui::Combo("##ExposureMode", &exposureModeIndex, exposureModeEnum.first, exposureModeEnum.second))
    {
        CEnumEntryPtr entry = config_->exposureMode->GetEntryByName(exposureModeEnum.first[exposureModeIndex]);
        config_->exposureMode->SetIntValue(entry->GetValue());
    }
    ImGui::Text("Exposure"); ImGui::SameLine(120);
    if(ImGui::InputFloat("##Exposure", &exposureTime) && IsWritable(config_->exposureTime))
    {
        IGNORE_SPINNAKER_ERROR(config_->exposureTime->SetValue(exposureTime));
    }

    ImGui::Separator();
    float gain = config_->gain->GetValue();
    int gainModeIndex = config_->gainMode->GetCurrentEntry()->GetValue();
    ImGui::Text("Gain Mode"); ImGui::SameLine(120);
    if(ImGui::Combo("##GainMode", &gainModeIndex, gainModeEnum.first, gainModeEnum.second))
    {
        CEnumEntryPtr entry = config_->gainMode->GetEntryByName(gainModeEnum.first[gainModeIndex]);
        config_->gainMode->SetIntValue(entry->GetValue());
    }

    ImGui::Text("Gain"); ImGui::SameLine(120);
    if(ImGui::InputFloat("##Gain", &gain) && IsWritable(config_->gain))
    {
        IGNORE_SPINNAKER_ERROR(config_->gain->SetValue((double)gain));
    }

    ImGui::Separator();
    
    int triggerModeIndex = config_->triggerMode->GetCurrentEntry()->GetValue();
    int triggerSourceIndex = config_->triggerSource->GetCurrentEntry()->GetValue();
    ImGui::Text("Trigger Mode"); ImGui::SameLine(120); 
    if(ImGui::Combo("##TriggerMode", &triggerModeIndex, triggerModeEnum.first, triggerModeEnum.second))
    {
        CEnumEntryPtr entry = config_->triggerMode->GetEntryByName(triggerModeEnum.first[triggerModeIndex]);
        config_->triggerMode->SetIntValue(entry->GetValue());
    }

    ImGui::Text("Trigger Source"); ImGui::SameLine(120);
    if(ImGui::Combo("##TriggerSource", &triggerSourceIndex, triggerSourceEnum.first, triggerSourceEnum.second) && IsWritable(config_->triggerSource))
    {
        CEnumEntryPtr entry = config_->triggerSource->GetEntryByName(triggerSourceEnum.first[triggerSourceIndex]);
        IGNORE_SPINNAKER_ERROR(config_->triggerSource->SetIntValue(entry->GetValue()));
    }

    ImGui::Separator();
    if (ImGui::SmallButton("Show Info"))
        showInfoWindow = true;
    
    ImGui::SameLine();
    if(ImGui::SmallButton("Start"))
    {
        exit(0);
    }
    
    if (showInfoWindow)
    {
        ImGui::Begin("Info", &showInfoWindow, ImGuiWindowFlags_NoCollapse); // Pass the bool to let user close it
        ImGui::Text("Max Width: %ld, Min Width %ld",config_->width->GetMax(),config_->width->GetMin());
        ImGui::Text("Max Height: %ld, Min Height: %ld",config_->height->GetMax(),config_->height->GetMin());
        ImGui::Text("Max Framerate: %f, Min Framerate: %f",config_->frameRate->GetMax(),config_->frameRate->GetMin());
        ImGui::Text("Max Exposure Time: %f, Min Exposure Time: %f",config_->exposureTime->GetMax(),config_->exposureTime->GetMin());
        ImGui::Text("Max Gain: %f, Min Gain: %f",config_->gain->GetMax(),config_->gain->GetMin());
        ImGui::End();
    }
    ImGui::End();
}

MainWindow::MainWindow(std::shared_ptr<FLIRCamera::Config> config):
    WindowBase(width*2, height, "PhaseVisualizer")
{
    width = config->width->GetValue();
    height = config->height->GetValue();

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &frameTexture);
    glBindTexture(GL_TEXTURE_2D, frameTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &phaseTexture);
    glBindTexture(GL_TEXTURE_2D, phaseTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
}

MainWindow::~MainWindow()
{

}

void MainWindow::update(void* frameData, void* phaseData)
{
    glBindTexture(GL_TEXTURE_2D, frameTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_LUMINANCE, GL_UNSIGNED_BYTE, frameData);

    if(phaseData != nullptr)
    {
        glBindTexture(GL_TEXTURE_2D, phaseTexture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, phaseData);
    }
}

void MainWindow::render()
{   
    ImGui::Begin("Main Panel", nullptr, ImGuiWindowFlags_AlwaysAutoResize
    | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse
    | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus ); 

    now = std::chrono::system_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
    last = now;
    ImVec2 avail = ImGui::GetContentRegionAvail();
    float leftWidth = avail.x * 0.20;
    float rightWidth = avail.x * 0.80;
    
    ImGui::BeginChild("Left Panel", ImVec2(leftWidth, avail.y),true);

    // Dropdown
    ImGui::Text("Actual FrameRate: %f",(1000.0 / duration));//Measure actual framerate
    ImGui::Text("Choose Phase Algorithm");
    const char* algorithms[] = { "Novak", "Four Point", "Carre"};
    int selectedAlgorithm;//static_cast<int>(algorithm.load());
    if(ImGui::Combo("##AlgorithmDropdown", &selectedAlgorithm, algorithms, IM_ARRAYSIZE(algorithms)))
    {
        //algorithm = static_cast<GPU::PhaseAlgorithm>(selectedAlgorithm);
        std::cout << "Currently Selected Algorithm: " << selectedAlgorithm << std::endl;
    }

    ImGui::Separator();
    ImGui::Text("Number of Successive Images"); ImGui::SameLine(120);
    ImGui::InputInt("##numSuccessiveImages", &numSuccessiveImages, 1);
    if (ImGui::Button("Save Phase Maps")) 
    {
    /*  std::shared_ptr<float> phaseMap = std::get<2>(tuple);
        if(phaseMap != nullptr)
        {
            cv::Mat phaseMat(height, width, CV_32F);
            if(cudaMemcpy(phaseMat.data, phaseMap.get(), width*height*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                std::cout << "Failed to copy data" << std::endl;
            else
                writeMatToCSV(phaseMat, "/home/nvidia/images/" + std::to_string(imageCount) + ".csv");
        }
    */
 
    }

    //ImGui::SameLine();
    ImGui::EndChild();

    ImGui::SameLine();
    // ImGui::Spacing();
    ImGui::BeginChild("Right Panel", ImVec2(rightWidth, avail.y), true, ImGuiWindowFlags_HorizontalScrollbar);

    ImGui::Image((ImTextureID)frameTexture, ImVec2(width, height));
    ImGui::SameLine();
    ImGui::Image((ImTextureID)phaseTexture, ImVec2(width, height));

    ImGui::EndChild();
    ImGui::End();
}

void MainWindow::writeMatToCSV(const cv::Mat mat, const std::string& fileName){
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

// ImGui::Begin("Main Panel", nullptr, ImGuiWindowFlags_AlwaysAutoResize
//     | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse
//     | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus ); 


//     renderer();

//     ImGui::End();

// void mainUI(MainParameters& config_->) 
// {
//     ImGui::Begin("Side Panel", nullptr, ImGuiWindowFlags_NoResize); 

//     // Dropdown
//     ImGui::Text("Choose Phase Algorithm");    
//     ImGui::Combo("##AlgorithmDropdown", &config_->.algorithmIndex, config_->.algorithms, config_->.nAlgorithms);

//     ImGui::Separator();
    
//     ImGui::InputInt("Successive Images", &config_->.nSavedImages);

//     if (ImGui::Button("Save Phase Maps")) 
//     {
//         config_->.onSaveImage();
//     }

//     ImGui::End();

//     ImGui::SameLine();


// }