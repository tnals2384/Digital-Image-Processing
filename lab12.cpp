#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <vector>
#include <cmath>

#define _USE_MATH_DEFINES

using namespace std;
using namespace cv;

//영상 노출 시간을 불러오는 함수
void readImagesAndTimes(vector<Mat>& images, vector<float>& times) {
    int numImages = 4;
    static const float timesArray[] = { 1 / 30.0f, 0.25f, 2.5f, 15.0f };
    times.assign(timesArray, timesArray + numImages); 
    //static const char* filenames[] = { "img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg" }
    static const char* filenames[] = { "0.01.jpg", "0.033.jpg", "0.2.jpg", "1.jpg" };
    for (int i = 0; i < numImages; i++) {
        Mat im = imread(filenames[i]);
        images.push_back(im);
    }
}

Mat GetHistogram(Mat& src) {
    Mat histogram;
    const int* channel_numbers = { 0 };
    float channel_range[] = { 0.0, 255.0 };
    const float* channel_ranges = channel_range;
    int number_bins = 255;

    calcHist(&src, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);

    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / number_bins);

    Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
    normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < number_bins; i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(histogram.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);
    }

    return histImage;
}

void ex1() {
    cout << "Reading images and exposure times ... \n";
    vector<Mat> images;
    vector<Mat> gray_images;
    vector<float> times;
    readImagesAndTimes(images, times);  //노출 시간 불러오기
    cout << "finished\n";

 //   static const char* filenames[] = { "img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg" };
    static const char* filenames[] = { "0.01.jpg", "0.033.jpg", "0.2.jpg", "1.jpg" };

    //입력 영상들을 grayscale로 불러와 histogram 출력
    for (int i = 0; i < 4; i++) {
        Mat im = imread(filenames[i],0);
        gray_images.push_back(im);
    }
    imshow("h1",GetHistogram(gray_images[0]));
    imshow("h2", GetHistogram(gray_images[1]));
    imshow("h3", GetHistogram(gray_images[2]));
    imshow("h4", GetHistogram(gray_images[3]));

    //영상 정렬
    cout << "Aligning images ...\n";
    Ptr<AlignMTB> alignMTB = createAlignMTB(); //median threshold를 이용
    alignMTB->process(images, images);

    //camera fesponse fuction 복원
    cout << "calculating Camera Response Function ...\n";
    Mat  responseDebevec;
    Ptr<CalibrateDebevec> calibrateDebevec = createCalibrateDebevec();
    calibrateDebevec->process(images, responseDebevec, times);
    cout << "----- CRF -----\n";
    cout << responseDebevec << "\n";

    //24bit 표현 범위로 영상 병합
    cout << "Merging images into one HDR images ... \n";
    Mat hdrDebevec;
    Ptr<MergeDebevec> mergeDebevec = createMergeDebevec();
    mergeDebevec->process(images, hdrDebevec, times, responseDebevec);
    cout << "saved hdrDebevec.hdr\n";

    //drago tone mappting을 이용하여 24bit -> 8bit로 표현
    cout << "Tonemapping using Drago's method ...\n";
    Mat ldrDrago;
    Ptr<TonemapDrago> tonemapDrago = createTonemapDrago(1.0f, 0.7f, 0.85f);
    tonemapDrago->process(hdrDebevec, ldrDrago);
    ldrDrago = 3 * ldrDrago;
    
    //결과 이미지 저장 및 결과이미지의 histogram 출력
    imwrite("ldr-Drago.jpg", ldrDrago * 255);
    Mat drago = ldrDrago * 255;
    cvtColor(drago, drago, COLOR_RGB2GRAY);
    imshow("drago's his", GetHistogram(drago));
    cout << "saved ldr-Drago.jpg\n";
    waitKey();
}

int main() {
    ex1();

    return 0;
}

/*cout << "Tonemapping using Reinhard's method ... \n";
Mat ldrReinhard;
Ptr<TonemapReinhard> tonemapReinhard = createTonemapReinhard(1.5f, 0, 0, 0);
tonemapReinhard->process(hdrDebevec, ldrReinhard);
imwrite("ldr-Reinharad.jpg", ldrReinhard * 255);
cout << "saved ldr-Reinhard.jpg\n";

cout << "Tonemapping using Mantiuk's method ... \n";
Mat ldrMantiuk;
Ptr<TonemapMantiuk> tonemapMantiuk = createTonemapMantiuk(2.2f, 0.85f, 1.2f);
tonemapMantiuk->process(hdrDebevec, ldrMantiuk);
ldrMantiuk = 3 * ldrMantiuk;
imwrite("ldr-Mantiuk.jpg", ldrMantiuk * 255);
cout << "saved ldr-Mantiuk.jpg\n";*/