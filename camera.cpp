#include <opencv2/opencv.hpp>
#include <iostream>

int main() {

    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "camera failed\n";
        return 1;
    }

    std::cout << "camera preview running\n";

    while (true) {

        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            continue;
        }

        cv::imshow("camera", frame);

        if (cv::waitKey(30) == 27) {
            break;
        }
    }

    cap.release();
    return 0;
}