#include "FLIRCamera.h"
#include "GPU.h"
#include <memory>
#include <chrono>
#include <queue>
#include <fstream>
#include <algorithm>
#include <cctype>

#include <pthread.h>

#include <qmetatype.h>
#include <qdatastream.h>

#include <opencv2/opencv.hpp>

#include <boost/program_options.hpp>


#include <QApplication>
#include <QMainWindow>
#include <QWindow>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLTexture>
#include <QOpenGLBuffer>

#define WIDTH 720
#define HEIGHT 540
#define FRAMERATE 60.0
#define EXPOSURE 12
#define GAIN 0.0


std::deque<Spinnaker::ImagePtr> imageQueue1;
pthread_mutex_t imageQueue1Mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t imageQueue1Cond = PTHREAD_COND_INITIALIZER;


std::deque<std::tuple<Spinnaker::ImagePtr, std::shared_ptr<uint8_t>, std::shared_ptr<float>>> imageQueue2;
pthread_mutex_t imageQueue2Mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t imageQueue2Cond = PTHREAD_COND_INITIALIZER;

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

// namespace po = boost::program_options;

// int main(int argc, char* argv[])
// {
//     // po::options_description desc("CHPC options");
//     // desc.add_options()
//     //     ("fps", po::value<int>()->default_value(FRAMERATE), "Set the frame rate")
//     //     ("width", po::value<int>()->default_value(WIDTH), "Set the frame width")
//     //     ("height", po::value<int>()->default_value(HEIGHT), "Set the frame height")
//     //     ("exposure", po::value<int>()->default_value(EXPOSURE), "Set the exposure time(us)")
//     //     ("gain", po::value<double>()->default_value(GAIN), "Set the gain")
//     //     ("algorithm", po::value<std::string>()->default_value("Novak", "Set the algorithm to create"))
//     // ;

//     // po::variables_map vm;
//     // po::store(po::parse_command_line(argc, argv, desc), vm);
//     // po::notify(vm);

//     // int width = vm["width"].as<int>();
//     // int height = vm["height"].as<int>();
//     // int frameRate = vm["fps"].as<int>();
//     int triggerLine = 1;
//     // int exposure = vm["exposure"].as<int>();
//     // double gain = vm["gain"].as<double>();
//     // std::string algorithmStr = vm["algorithm"].as<std::string>();
//     // std::transform(algorithmStr.begin(),algorithmStr.end(), algorithmStr.begin(), [](char c){return std::toupper(c);});
//     // GPU::PhaseAlgorithm algorithm;
//     // if(algorithmStr == "NOVAK")
//     //     algorithm = GPU::PhaseAlgorithm::NOVAK;
//     // else if(algorithmStr == "FOURPOINT")
//     //     algorithm = GPU::PhaseAlgorithm::FOURPOINT;
//     // else if(algorithmStr == "CARRE")
//     //     algorithm = GPU::PhaseAlgorithm::CARRE;
//     // else
//     // {
//     //     std::cout << "Invaild algorithm: " << algorithmStr << std::endl;
//     //     return -1; 
//     // }
        
//     // switch(argc)
//     // {
//     //     case 5:
//     //     triggerLine = std::stoi(argv[4]);
//     //     case 4:
//     //     frameRate = std::stoi(argv[3]);
//     //     case 3:
//     //     height = std::stoi(argv[2]);
//     //     case 2:
//     //     width = std::stoi(argv[1]);
//     //     break;
//     //     default:
//     //     break;
//     // }

//     if(!glfwInit())
//     {
//         std::cout << "Failed to init glfw library" << std::endl;
//         return -1;
//     }

//     int width = WIDTH;
//     int height = HEIGHT;
//     int exposure = EXPOSURE;
//     int frameRate = FRAMERATE;
//     double gain = GAIN;

//     GPU gpu(width,height,50);
//     FLIRCamera cam;
//     cam.open(0);
//     cam.setResolution(width,height);
//     cam.setFPS(frameRate);
//     cam.setExposureTime(exposure);
//     cam.setPixelFormat("Mono16");
//     cam.setGain(gain);

//     if(triggerLine != -1)
//     {
//         cam.enableTrigger(TriggerSource_Line3);
//     }

//     pthread_t gpuThread;
//     if(pthread_create(&gpuThread, NULL, gpuThreadFunc, &gpu) == -1)
//     {
//         std::cout << "Failed to create GPU thread" << std::endl;
//         return 1;
//     }

//     pthread_t cameraThread;
//     if(pthread_create(&cameraThread, NULL, cameraThreadFunc, &cam) == -1)
//     {
//         std::cout << "Failed to create camera thread" << std::endl;
//         return 1;
//     }

//     glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

//     // int key = GLFW_KEY_UNKNOWN;
//     GLFWwindow* frame = glfwCreateWindow(width * 2, height, "frame", NULL, NULL);
//     // glfwSetWindowUserPointer(frame, &key);
//     glfwMakeContextCurrent(frame);
//     if(!frame)
//     {
//         std::cout << "Failed to create the main window" << std::endl;
//     }

//     GLuint frameTexture;
//     glEnable(GL_TEXTURE_2D);
//     glGenTextures(1, &frameTexture);
//     glBindTexture(GL_TEXTURE_2D, frameTexture);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

//     glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE16, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_SHORT , NULL);

//     GLuint phaseTexture;
//     glEnable(GL_TEXTURE_2D);
//     glGenTextures(1, &phaseTexture);
//     glBindTexture(GL_TEXTURE_2D, phaseTexture);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

//     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

//     auto last = std::chrono::system_clock::now();
//     bool spacePressed = false;
//     int imageCount = 0;

//     while(!glfwWindowShouldClose(frame))
//     {
//         glfwPollEvents();
        
//         std::tuple<Spinnaker::ImagePtr, std::shared_ptr<uint8_t>, std::shared_ptr<float>> tuple;

//         pthread_mutex_lock(&imageQueue2Mutex);
//         while(imageQueue2.size() == 0)
//             pthread_cond_wait(&imageQueue2Cond, &imageQueue2Mutex);

//         tuple = std::move(imageQueue2.back());
//         imageQueue2.clear();

//         pthread_mutex_unlock(&imageQueue2Mutex);

//         glBindTexture(GL_TEXTURE_2D, frameTexture);
//         glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_LUMINANCE, GL_UNSIGNED_SHORT, std::get<0>(tuple)->GetData());
//         glBegin(GL_QUADS);
//         glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
//         glTexCoord2f(1.0, 0.0); glVertex2f(0.0, -1.0);
//         glTexCoord2f(1.0, 1.0); glVertex2f(0.0, 1.0);
//         glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
//         glEnd();

//         if(std::get<1>(tuple) != nullptr)
//         {
//             glBindTexture(GL_TEXTURE_2D, phaseTexture);
//             glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, std::get<1>(tuple).get());
//             glBegin(GL_QUADS);
//             glTexCoord2f(0.0, 0.0); glVertex2f(0.0, -1.0);
//             glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
//             glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
//             glTexCoord2f(0.0, 1.0); glVertex2f(0.0, 1.0);
//             glEnd();

//             cv::Mat();
//         }
//         glfwSwapBuffers(frame);

        
//         int state = glfwGetKey(frame, GLFW_KEY_SPACE);
//         if(state == GLFW_PRESS && !spacePressed)
//         {
//             spacePressed = true;
//         }
//         if(state == GLFW_RELEASE && spacePressed)
//         {
//             spacePressed = false;
//             imageCount ++;
//             // if(!cv::imwrite("/home/nvidia/images/" + std::to_string(imageCount) + ".png", cv::Mat(height, width, CV_32F, std::get<2>(tuple).get())))
//             //     std::cout << "Failed to save image" << std::endl;
//             // else
//             //     std::cout << "Image saved" << std::endl;
//             std::shared_ptr<float> phaseMap = std::get<2>(tuple);
//             if(phaseMap != nullptr)
//             {
//                 cv::Mat phaseMat(height, width, CV_32F);
//                 if(cudaMemcpy(phaseMat.data, phaseMap.get(), width*height*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
//                     std::cout << "Failed to copy data" << std::endl;
//                 else
//                     writeMatToCSV(phaseMat, "/home/nvidia/images/" + std::to_string(imageCount) + ".csv");
//             }
//         }
        

//         auto now = std::chrono::system_clock::now();
//         int duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
//         // std::cout << "Frame rate: " << 1000.0 / duration << std::endl;
//         last = now;

//     }

//     pthread_cancel(gpuThread);
//     pthread_cancel(cameraThread);

//     pthread_join(gpuThread, NULL);
//     pthread_join(cameraThread, NULL);

//     imageQueue1.clear();
//     imageQueue2.clear();

//     cam.close();

//     glfwDestroyWindow(frame);
//     glfwTerminate();

//     return 0;
// }


class PhaseEvent: public QEvent
{
    public:
    using Data = std::tuple<Spinnaker::ImagePtr, std::shared_ptr<uint8_t>, std::shared_ptr<float>>;
    static const QEvent::Type EventType;

    PhaseEvent(Data&& data):
        QEvent(EventType),
        data_(std::move(data))
    {

    }
    
    Data data_;
};

const QEvent::Type PhaseEvent::EventType = static_cast<QEvent::Type>(QEvent::registerEventType());

class OpenGLWindow : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT
    public:
    explicit OpenGLWindow(QWindow* parent = nullptr){}

    ~OpenGLWindow()
    {
        delete frameTexture;
        delete phaseTexture;
        delete buffer;
    }

    protected:
    void initializeGL() override
    {
        initializeOpenGLFunctions();
        glEnable(GL_TEXTURE_2D);

        frameTexture = new QOpenGLTexture(QOpenGLTexture::Target2D);
        frameTexture -> setSize(WIDTH, HEIGHT);
        frameTexture -> setMinificationFilter(QOpenGLTexture::Linear);
        frameTexture -> setMagnificationFilter(QOpenGLTexture::Linear);
        frameTexture -> allocateStorage(QOpenGLTexture::Luminance, QOpenGLTexture::UInt16);

        phaseTexture = new QOpenGLTexture(QOpenGLTexture::Target2D);
        phaseTexture -> setSize(WIDTH, HEIGHT);
        phaseTexture -> setMinificationFilter(QOpenGLTexture::Linear);
        phaseTexture -> setMagnificationFilter(QOpenGLTexture::Linear);
        phaseTexture -> allocateStorage(QOpenGLTexture::RGB, QOpenGLTexture::UInt32);

        std::vector<GLfloat> vertData{
            -1.0, -1.0, 0.0, 0.0,
            0.0, -1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0,
            -1.0, 1.0, 0.0, 1.0,

            0.0, -1.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            0.0, 1.0, 0.0, 1.0
        };

        buffer = new QOpenGLBuffer();
        buffer -> create();
        buffer -> bind();
        buffer -> allocate(vertData.data(), vertData.size()*sizeof(GLfloat));
        std::cout << "Open GL Initialized\n";
    }

    void resizeGL(int w, int h) override
    {
        paintGL();
    }

    void paintGL() override
    {
        glClearColor(1.0f,0.0f,0.0f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        frameTexture -> bind(0);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        phaseTexture -> bind(0);
        glDrawArrays(GL_TRIANGLE_STRIP, 4, 4);
    }

    private:

    QOpenGLTexture *frameTexture = nullptr;
    QOpenGLTexture *phaseTexture = nullptr;
    QOpenGLBuffer *buffer;

    bool event(QEvent* event)
    {
        if(event->type() == PhaseEvent::EventType)
        {
            PhaseEvent* phaseEvent = reinterpret_cast<PhaseEvent*>(event);

            PhaseEvent::Data data = std::move(phaseEvent->data_);
            
            frameTexture -> setData(QOpenGLTexture::Luminance, QOpenGLTexture::UInt16, std::get<0>(data)->GetData());
            phaseTexture -> setData(QOpenGLTexture::RGB, QOpenGLTexture::UInt32, std::get<1>(data).get());

            update();

            return true;
        }

        return QOpenGLWidget::event(event);
    }
};

#include "main.moc"

// void* uiThreadFunc(void* arg)
// {
//     OpenGLWindow* w = reinterpret_cast<OpenGLWindow*>(arg);

//     while(true)
//     {
//         pthread_mutex_lock(&imageQueue2Mutex);
//         while(imageQueue2.size() == 0)
//             pthread_cond_wait(&imageQueue2Cond, &imageQueue2Mutex);
//         pthread_mutex_unlock(&imageQueue2Mutex);
//         //std::cout << "Image Queue Event!\n";
//         QMetaObject::invokeMethod(w, "update", Qt::QueuedConnection);
//         pthread_testcancel();
//     }

//     return 0;
// }


void gpuThreadCleanUp(void* arg)
{
    std::cout << "gpu thread exited" << std::endl;
}

void* gpuThreadFunc(void* arg)
{
    pthread_cleanup_push(gpuThreadCleanUp, NULL);

    std::pair<GPU*, OpenGLWindow*>* args = reinterpret_cast<std::pair<GPU*, OpenGLWindow*>*>(arg);
    GPU* gpu = args->first;
    OpenGLWindow* w = args->second;

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

        std::pair<std::shared_ptr<uint8_t>, std::shared_ptr<float>> pair = gpu->run(imagePtr, GPU::PhaseAlgorithm::CARRE);
        
        // pthread_mutex_lock(&imageQueue2Mutex);

        // imageQueue2.emplace_back(imagePtr, pair.first, pair.second);

        // pthread_cond_signal(&imageQueue2Cond);
        // QMetaObject::invokeMethod(w, "update", Qt::QueuedConnection);
        QApplication::postEvent(w, new PhaseEvent({imagePtr, pair.first, pair.second}));
        // pthread_mutex_unlock(&imageQueue2Mutex);

        pthread_testcancel();

        auto now = std::chrono::system_clock::now();
        int duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
        // std::cout << "Frame rate: " << 1000.0 / duration << std::endl;
        last = now;
    }
    pthread_cleanup_pop(1);
    return 0;
}


int main(int argc, char* argv[])
{
    std::string algorithmStr = "NOVAK";
    GPU::PhaseAlgorithm algorithm;
    if(algorithmStr == "NOVAK")
        algorithm = GPU::PhaseAlgorithm::NOVAK;
    else if(algorithmStr == "FOURPOINT")
        algorithm = GPU::PhaseAlgorithm::FOURPOINT;
    else if(algorithmStr == "CARRE")
        algorithm = GPU::PhaseAlgorithm::CARRE;
    else
    {
        std::cout << "Invaild algorithm: " << algorithmStr << std::endl;
        return -1; 
    }
    int width = WIDTH;
    int height = HEIGHT;
    int exposure = EXPOSURE;
    int frameRate = FRAMERATE;
    double gain = GAIN;
    int triggerLine = -1;

    GPU gpu(width,height,50);
    FLIRCamera cam;
    cam.open(0);
    cam.setResolution(width,height);
    cam.setFPS(frameRate);
    cam.setExposureTime(exposure);
    cam.setPixelFormat("Mono16");
    cam.setGain(gain);

    if(triggerLine == 1)
    {
        cam.enableTrigger(TriggerSource_Line3);
    }
    else{
        cam.disableTrigger();
    }

    pthread_t cameraThread;
    if(pthread_create(&cameraThread, NULL, cameraThreadFunc, &cam) == -1)
    {
        std::cout << "Failed to create camera thread" << std::endl;
        return 1;
    }

    QApplication app(argc, argv);
    OpenGLWindow w;

    // pthread_t uiThread;
    // if(pthread_create(&uiThread, NULL, uiThreadFunc, &w) == -1)
    // {
    //     std::cout << "Failed to create ui thread" << std::endl;
    //     return 1;
    // }

    
    // QSurfaceFormat format;
    // format.setSamples(16);
    pthread_t gpuThread;
    std::pair<GPU*, OpenGLWindow*> args = {&gpu, &w};
    if(pthread_create(&gpuThread, NULL, gpuThreadFunc, &args) == -1)
    {
        std::cout << "Failed to create GPU thread" << std::endl;
        return 1;
    }
    
    // w.setFormat(format);
    w.resize(width * 2, height);
    w.show();
    int ret = app.exec();

    // pthread_cancel(uiThread);
    pthread_cancel(gpuThread);
    pthread_cancel(cameraThread);

    // pthread_join(uiThread, NULL);
    pthread_join(gpuThread, NULL);
    pthread_join(cameraThread, NULL);

    imageQueue1.clear();
    imageQueue2.clear();

    cam.close();

    return ret;
}
