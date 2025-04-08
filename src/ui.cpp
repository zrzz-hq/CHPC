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

    ImGui::StyleColorsLight();  // Use Dark Mode Theme

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

// Initialize Dear ImGui
void InitImGui(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    ImGui::StyleColorsLight();  // Use Dark Mode Theme

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
    const char* algorithms[] = { "Carre", "Novak", "Four Point" };
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

    ImGui::Text("Input Device");
    const char* inputDevice[] = { "Camera 1" };
    static int selectedInputDevice = 0;
    ImGui::Combo("##DeviceDropdown", &selectedInputDevice, inputDevice, IM_ARRAYSIZE(inputDevice));
    ImGui::Separator();
    // Trigger and Save Buttons
    static bool triggerContinuous = false;
    ImGui::Checkbox("Trigger/Continuous", &triggerContinuous);

    static int triggerLine = 2, numSuccessiveImages = 1;
    const char* lineSelect[] = { "Trigger Line 1", "Trigger Line 2", "Trigger Line 3",  "Trigger Line 4"};
    ImGui::Combo("##TriggerDropdown", &triggerLine, lineSelect, IM_ARRAYSIZE(lineSelect));

    ImGui::Separator();
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