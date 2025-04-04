#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

// Initialize Dear ImGui
void InitImGui(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    ImGui::StyleColorsDark();  // Use Dark Mode Theme

    // Setup platform/renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

// Cleanup ImGui before closing
void CleanupImGui() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void RenderUI() {
    ImGui::Begin("Side Panel", nullptr, ImGuiWindowFlags_NoResize); 

    // Dropdown
    ImGui::Text("Choose Phase Algorithm");
    const char* algorithms[] = { "Algorithm 1", "Algorithm 2", "Algorithm 3" };
    static int selectedAlgorithm = 0;
    ImGui::Combo("##AlgorithmDropdown", &selectedAlgorithm, algorithms, IM_ARRAYSIZE(algorithms));

    // Parameters Section
    ImGui::Separator();
    ImGui::Text("Parameters");

    static int width = 1920, height = 1080;
    ImGui::InputInt("Resolution Width", &width);
    ImGui::InputInt("Resolution Height", &height);

    static float framerate = 60.0f, exposure = 0.5f, gain = 1.0f;
    ImGui::InputFloat("Framerate", &framerate);
    ImGui::InputFloat("Exposure", &exposure);
    ImGui::InputFloat("Gain", &gain);

    static char inputDevice[128] = "Camera 1";
    ImGui::InputText("Input Device", inputDevice, IM_ARRAYSIZE(inputDevice));

    // Trigger and Save Buttons
    static bool triggerContinuous = false;
    ImGui::Checkbox("Trigger/Continuous", &triggerContinuous);

    static int triggerLine = 0, numSuccessiveImages = 1;
    ImGui::InputInt("Trigger Line", &triggerLine);
    ImGui::InputInt("Successive Images", &numSuccessiveImages);

    if (ImGui::Button("Save Phase Maps")) {
        // Handle button click
    }

    ImGui::Separator();
    
    // Start & Stop Buttons
    if (ImGui::Button("Start", ImVec2(80, 40))) {
        // Handle start
    }
    ImGui::SameLine();
    if (ImGui::Button("Stop", ImVec2(80, 40))) {
        // Handle stop
    }

    ImGui::End();
}