#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "ui.h"

UI::UI(GLFWwindow* window, std::function<void()> renderer):renderer(renderer)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    //io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    (void)io;

    ImGui::StyleColorsLight();  // Use Light Mode Theme

    // Setup platform/renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    // ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
}

UI::~UI()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void UI::render()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);


    ImGui::Begin("Main Panel", nullptr, ImGuiWindowFlags_AlwaysAutoResize
    | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse
    | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus ); 


    renderer();

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void errorUI(bool& shouldClose)
{
    ImGui::Text("Error: No Camera Detected");
    if (ImGui::Button("Close Window")){
        shouldClose = true;
    }
}

void startupUI(StartupParameters& params)
{
    // ImGui::Begin("StartupPanel", nullptr, ImGuiWindowFlags_NoResize); 
    ImGui::Text("Parameters");

    ImGui::InputInt("Width", &params.width);
    ImGui::InputInt("Height", &params.height);

    ImGui::InputInt("Framerate", &params.frameRate);
    ImGui::InputInt("Exposure", &params.exposureTime);
    ImGui::InputFloat("Gain", &params.gain);

    ImGui::Text("Input Device");

    ImGui::Combo("##DeviceDropdown", &params.deviceIndex, params.deviceNames, params.nDevices);
    ImGui::Separator();
    
    static bool triggerContinuous = false;
    ImGui::Checkbox("Trigger/Continuous", &triggerContinuous);

    static int triggerLine = 2;
    const char* lineSelect[] = { "Trigger Line 1", "Trigger Line 2", "Trigger Line 3",  "Trigger Line 4"};
    ImGui::Combo("##TriggerDropdown", &triggerLine, lineSelect, IM_ARRAYSIZE(lineSelect));


}

void mainUI(MainParameters& params) 
{
    ImGui::Begin("Side Panel", nullptr, ImGuiWindowFlags_NoResize); 

    // Dropdown
    ImGui::Text("Choose Phase Algorithm");    
    ImGui::Combo("##AlgorithmDropdown", &params.algorithmIndex, params.algorithms, params.nAlgorithms);

    ImGui::Separator();
    
    ImGui::InputInt("Successive Images", &params.nSavedImages);

    if (ImGui::Button("Save Phase Maps")) 
    {
        params.onSaveImage();
    }

    ImGui::End();

    ImGui::SameLine();


}