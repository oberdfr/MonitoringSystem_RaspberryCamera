#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <thread>
#include "yolo-fastestv2.h"
#include "include/HTTPRequest.hpp"

#include <nadjieb/mjpeg_streamer.hpp>

yoloFastestv2 yoloF2;

// for convenience
using MJPEGStreamer = nadjieb::MJPEGStreamer;

const char* class_names[] = {
    "background", "person", "bicycle",
    "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
    "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
};

float round(int n){
    float y, d;
    y= n*100;
    d = y -(int)y;
    y= (float)(int)(n*100)/100;
    if (d > 0.5)
        y += 0.01;
    return y;
}

static int draw_objects(cv::Mat& cvImg, const std::vector<TargetBox>& boxes, bool display_all) {
    int nPeople = 0;

    for (size_t i = 0; i < boxes.size(); i++) {
        if (!display_all && boxes[i].cate != 0) {
            continue;
        }

        // increment people counter
        nPeople++;

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[boxes[i].cate + 1], boxes[i].score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = boxes[i].x1;
        int y = boxes[i].y1 - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > cvImg.cols) x = cvImg.cols - label_size.width;

        cv::rectangle(cvImg, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(cvImg, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        cv::rectangle(cvImg, cv::Point(boxes[i].x1, boxes[i].y1),
                      cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255, 0, 0));
    }

    return nPeople;
}

void send_http_request(float people_count) {
    try {
        // you can pass http::InternetProtocol::V6 to Request to make an IPv6 request
        http::Request request{"http://192.168.100.103:5000/mandacounter?people=" + std::to_string(round(people_count))};

        // send a get request
        const auto response = request.send("GET");
        std::cout << std::string{response.body.begin(), response.body.end()} << '\n'; // print the result
    }
    catch (const std::exception& e) {
        std::cerr << "Request failed, error: " << e.what() << '\n';
    }
}

int main(int argc, char** argv)
{
    int counter = 0;
    float FPS[16];
    int i;
    cv::Mat frame;
    //some timing
    std::chrono::steady_clock::time_point Tbegin, Tend;

    //number of people
    int nPeopleNow = 0;
    float nPeopleBuffer = 0;
    bool display_all = false; // Variabile toggle per visualizzare tutte le categorie o solo le persone

    for (i = 0; i < 16; i++) FPS[i] = 0.0;

    yoloF2.init(false); //we have no GPU

    yoloF2.loadModel("yolo-fastestv2-opt.param", "yolo-fastestv2-opt.bin");

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Unable to open the camera" << std::endl;
        return 0;
    }

    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};

    MJPEGStreamer streamer;

    // By default 1 worker is used for streaming
    // if you want to use 4 workers:
    //      streamer.start(8080, 4);
    streamer.start(8000);

    std::cout << "Start grabbing, press ESC on Live window to terminate" << std::endl;

    while (streamer.isRunning()) {
        // frame = cv::imread("000139.jpg");  //need to refresh frame before dnn class detection
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "ERROR: Unable to grab from the camera" << std::endl;
            exit(EXIT_FAILURE);
        }

        Tbegin = std::chrono::steady_clock::now();

        std::vector<TargetBox> boxes;
        yoloF2.detection(frame, boxes);
        nPeopleNow = draw_objects(frame, boxes, display_all); // Passa display_all a draw_objects
        nPeopleBuffer = nPeopleBuffer + nPeopleNow;
        Tend = std::chrono::steady_clock::now();

        //calculate frame rate
        //f = std::chrono::duration_cast<std::chrono::milliseconds>(Tend - Tbegin).count();
        //if (f > 0.0) FPS[(Fcnt++ & 0x0F)] = 1000.0 / f;
        //for (f = 0.0, i = 0; i < 16; i++) { f += FPS[i]; }
        //putText(frame, cv::format("FPS %0.2f", f / 16), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255));

        // Show number of people
        //putText(frame, cv::format("People: %d", nPeople), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255));

        // http://localhost:8000/bgr
        std::vector<uchar> buff_bgr;
        cv::imencode(".jpg", frame, buff_bgr, params);
        streamer.publish("/bgr", std::string(buff_bgr.begin(), buff_bgr.end()));

        if (counter >= 60){  // Riduci la frequenza di invio delle richieste HTTP
            std::cout << "counter reached 60: starting http request" << std::endl;
            std::cout << "counter resetted to 0" << std::endl;
            //httpreq
            std::thread http_thread(send_http_request, nPeopleBuffer/counter);
            http_thread.detach();
            counter = 0;
            nPeopleBuffer = nPeopleNow;

        } else {
            counter++;
        }
    }

    streamer.stop();
    return 0;
}
