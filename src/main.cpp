#include "FLIRCamera.h"
#include "GPU.h"
#include <memory>
#include <chrono>
#include <queue>

#include <pthread.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <GL/gl.h>
#include <GL/glx.h>

#define WIDTH 800
#define HEIGHT 600
#define FRAMERATE 60.0


std::deque<Spinnaker::ImagePtr> imageQueue1;
pthread_mutex_t imageQueue1Mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t imageQueue1Cond = PTHREAD_COND_INITIALIZER;


std::deque<std::pair<Spinnaker::ImagePtr, std::shared_ptr<uint8_t>>> imageQueue2;
pthread_mutex_t imageQueue2Mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t imageQueue2Cond = PTHREAD_COND_INITIALIZER;

void gpuThreadCleanUp(void* arg)
{
    std::cout << "gpu thread exited" << std::endl;
}

void* gpuThreadFunc(void* arg)
{
    pthread_cleanup_push(gpuThreadCleanUp, NULL);
    GPU* gpu = reinterpret_cast<GPU*>(arg);

    gpu->getCudaVersion();
    while(1)
    {
        Spinnaker::ImagePtr imagePtr;
        pthread_mutex_lock(&imageQueue1Mutex);

        while(imageQueue1.size() == 0)
            pthread_cond_wait(&imageQueue1Cond, &imageQueue1Mutex);

        imagePtr = imageQueue1.back();
        imageQueue1.clear();

        pthread_mutex_unlock(&imageQueue1Mutex);

        std::shared_ptr<uint8_t> cosine = gpu->runNovak(imagePtr);
        
        pthread_mutex_lock(&imageQueue2Mutex);

        imageQueue2.emplace_back(std::make_pair(imagePtr,cosine));

        pthread_cond_signal(&imageQueue2Cond);
        pthread_mutex_unlock(&imageQueue2Mutex);

        pthread_testcancel();
    }
    pthread_cleanup_pop(1);
    return 0;
}

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

int main(int argc, char* argv[])
{
    int width = WIDTH;
    int height = HEIGHT;
    int frameRate = FRAMERATE;
    switch(argc)
    {
        case 4:
        frameRate = std::stoi(argv[3]);
        case 3:
        height = std::stoi(argv[2]);
        case 2:
        width = std::stoi(argv[1]);
        break;
        default:
        break;
    }
    

    GPU gpu(width,height,50);
    FLIRCamera cam;
    cam.open(0);
    cam.setResolution(width,height);
    cam.setFPS(frameRate);

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

    Display * display = XOpenDisplay(NULL);
    if (!display)
    {
        std::cout << "Failed to open X display" << std::endl;
        return -1;
    }

    int screen = DefaultScreen(display);
    Window root = RootWindow(display, screen);
    GLint glAttribs[] = {GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None};
    XVisualInfo* vi = glXChooseVisual(display, screen, glAttribs);
    if (!vi)
    {
        std::cout << "No suitable OpenGL visual found" << std::endl;
        return -1;
    }

    Colormap cmap = XCreateColormap(display, root, vi->visual, AllocNone);
    
    XSetWindowAttributes swa;
    swa.colormap = cmap;
    swa.event_mask = ExposureMask | KeyPressMask;

    Window frame = XCreateWindow(display, root, 0, 0, width, height, 0, vi->depth, InputOutput, vi->visual, CWColormap | CWEventMask, &swa);
    // XSetWindowColormap(display, frame, cmap);
    XStoreName(display, frame, "frame");
    XMapWindow(display, frame);

    Window phase = XCreateWindow(display, root, 0, 0, width, height, 0, vi->depth, InputOutput, vi->visual, CWColormap | CWEventMask, &swa);
    XStoreName(display, phase, "phase");
    XMapWindow(display, phase);

    Atom wmDelete = XInternAtom(display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(display, frame, &wmDelete, 1);
    XSetWMProtocols(display, phase, &wmDelete, 1);

    GLXContext glc = glXCreateContext(display, vi, NULL, GL_TRUE);
    glXMakeCurrent(display, frame, glc);

    GLuint frameTexture;
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &frameTexture);
    glBindTexture(GL_TEXTURE_2D, frameTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);

    glXMakeCurrent(display, phase, glc);
    GLuint phaseTexture;
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &phaseTexture);
    glBindTexture(GL_TEXTURE_2D, phaseTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);

    XEvent event;
    auto last = std::chrono::system_clock::now();
    while(1)
    {
        if(XPending(display) > 0)
        {
            XNextEvent(display, &event);

            if(event.type == ClientMessage)
            {
                if((Atom)event.xclient.data.l[0] == wmDelete)
                    break;
            }

        }
        
        std::pair<Spinnaker::ImagePtr, std::shared_ptr<uint8_t>> imagePair;

        pthread_mutex_lock(&imageQueue2Mutex);
        while(imageQueue2.size() == 0)
            pthread_cond_wait(&imageQueue2Cond, &imageQueue2Mutex);

        imagePair = std::move(imageQueue2.back());
        imageQueue2.clear();

        pthread_mutex_unlock(&imageQueue2Mutex);

        auto now = std::chrono::system_clock::now();
        int duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
        last = now;

        glXMakeCurrent(display, frame, glc);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindTexture(GL_TEXTURE_2D, frameTexture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_LUMINANCE, GL_UNSIGNED_BYTE, imagePair.first->GetData());
        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
        glEnd();
        glXSwapBuffers(display, frame);

        if(imagePair.second != nullptr)
        {
            glXMakeCurrent(display, phase, glc);
            glClear(GL_COLOR_BUFFER_BIT);
            glBindTexture(GL_TEXTURE_2D, phaseTexture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_LUMINANCE, GL_UNSIGNED_BYTE, imagePair.second.get());
            glBegin(GL_QUADS);
            glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
            glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
            glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
            glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
            glEnd();
            glXSwapBuffers(display, phase);
        }

    }

    pthread_cancel(gpuThread);
    pthread_cancel(cameraThread);

    pthread_join(gpuThread, NULL);
    pthread_join(cameraThread, NULL);

    imageQueue1.clear();
    imageQueue2.clear();

    cam.close();

    glXDestroyContext(display, glc);
    XDestroyWindow(display, phase);
    XDestroyWindow(display, frame);
    XCloseDisplay(display);

    return 0;
}
