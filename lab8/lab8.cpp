#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;
using namespace cv;

void cvBlobDetection(Mat img) {
    SimpleBlobDetector::Params params;
    params.minThreshold = 2;
    params.maxThreshold = 300;
    params.filterByArea = true; 
    params.minArea = 10; //최소면적제한
    params.filterByCircularity = true; //원형에 얼마나 가까운지
    params.minCircularity = 0.55;
    params.filterByConvexity = true; //얼마나 볼록한지
    params.minConvexity = 0.1;


    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

    std::vector<KeyPoint> keypoints;
    detector->detect(img, keypoints);

    cout << keypoints.size() << "각형 " << endl; //몇각형인지 출력
}

void cvHarrisCorner() {
    Mat figure[4]; //삼각형,사각형,오각형,육각형 이미지를 읽어 figure배열에 넣기
    figure[0] = imread("triangle.png");
    figure[1] = imread("rect.png");
    figure[2] = imread("pentagon.png");
    figure[3] = imread("hexagon.png");

    for (int i = 0; i < 4; i++) { //4각형부터 육각형까지
        Mat img = figure[i]; 
        
        if (img.empty()) {
            cout << "Empty image!\n";
            exit(1);
        }

        resize(img, img, Size(500, 500), 0, 0, INTER_CUBIC); //500,500으로 resize

        Mat gray;
        cvtColor(img, gray, CV_BGR2GRAY); //흑백이미지로 변환

        Mat harr;
        cornerHarris(gray, harr, 2, 3, 0.04, BORDER_DEFAULT); //harriscorner
        normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat()); //정규화

        Mat harr_abs;
        convertScaleAbs(harr, harr_abs); //절댓값으로 변환

        //Print corners
        int thresh = 125;
        Mat result = img.clone();
        for (int y = 0; y < harr.rows; y += 1) {
            for (int x = 0; x < harr.cols; x += 1) {
                if ((int)harr.at<float>(y, x) > thresh)
                    circle(result, Point(x, y), 7, Scalar(255, 0, 255), 0, 4, 0); //코너부분을 원으로 표시
            }
        }
        cvBlobDetection(result); //blob detection 실행
        imshow("Source image", img);
        imshow("Harris image", harr_abs);
        imshow("Target image", result);
        waitKey(0);
    }
    destroyAllWindows();
}



void cvCoinDetection() {
    Mat img = imread("coin.png", IMREAD_COLOR);

    SimpleBlobDetector::Params params;
    params.minThreshold = 10; //최소 threshold값
    params.maxThreshold = 300; //최대 threshold값 
    params.filterByArea = true; //최소 최대 면적 제한
    params.minArea = 300; //최소
    params.maxArea = 10000; //최대
    params.filterByCircularity = true; //원형에 얼마나 가까운지 제한
    params.minCircularity = 0.65;
    params.filterByConvexity = true; //내부가 얼마나 볼록히 채워지는지 제한
    params.minConvexity = 0.9;
    params.filterByInertia = true; //타원이 원에 얼마나 가까운지 제한
    params.minInertiaRatio = 0.04;

    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params); //blob detector선언

    std::vector<KeyPoint> keypoints;
    detector->detect(img, keypoints); //이미지에서 keypoints 감지

    Mat result;
    //result이미지에 감지한 blob 그리기
    drawKeypoints(img, keypoints, result, Scalar(255, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("keypoints", result);

    cout << "동전의 개수: " << keypoints.size() << endl;// size()를 통해 원의 개수 출력
    waitKey(0);
    destroyAllWindows();
}



Mat warpPers(Mat src) {
    Mat dst;

    Point2f src_p[4], dst_p[4];
    src_p[0] = Point2f(0, 0);
    src_p[1] = Point2f(1200, 0);
    src_p[2] = Point2f(0, 800);
    src_p[3] = Point2f(1200, 800);

    dst_p[0] = Point2f(0, 0);
    dst_p[1] = Point2f(1200, 0);
    dst_p[2] = Point2f(0, 800);
    dst_p[3] = Point2f(1000, 700);

    Mat pers_mat = getPerspectiveTransform(src_p, dst_p);
    warpPerspective(src, dst, pers_mat, Size(1200, 800)); //src_p가 dst_p의 위치로 대응됨
    return dst;
}

void cvFeatureSIFT() {
    Mat img = imread("church.jpg", 1);
    // /*orginal sift*/
    Mat gray;
    cvtColor(img, gray, CV_BGR2GRAY); //grayscale로 변환
    Ptr<SiftFeatureDetector>detector = SiftFeatureDetector::create(); //detector선언
    std::vector<KeyPoint> keypoints;
    detector->detect(gray, keypoints); //keypoints감지

    Mat original;
    drawKeypoints(img, keypoints, original);
    imwrite("sift_original.jpg", original);
    imshow("Sift original", original); //original SIFT이미지 출력

    /* 투시변환 & 밝기변화 후 SIFT */
    img = warpPers(img); //투시변환 실행
    cvtColor(img, gray, CV_BGR2GRAY); //grayscale변환
    gray += 50; //밝기 50 올리기
    detector = SiftFeatureDetector::create();
    detector->detect(gray, keypoints);

    Mat result;
    drawKeypoints(img, keypoints, result);
    imwrite("sift_result.jpg", result);
    imshow("Sift result", result); //투시변환,밝기변환 후 SIFT이미지 출력
    waitKey(0);
    destroyAllWindows();
}


int main() {
    //cvCoinDetection(); 
    //cvHarrisCorner();
    cvFeatureSIFT();
    return 0;
}
