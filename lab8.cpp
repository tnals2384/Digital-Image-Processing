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
    params.minArea = 10; //�ּҸ�������
    params.filterByCircularity = true; //������ �󸶳� �������
    params.minCircularity = 0.55;
    params.filterByConvexity = true; //�󸶳� ��������
    params.minConvexity = 0.1;


    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

    std::vector<KeyPoint> keypoints;
    detector->detect(img, keypoints);

    cout << keypoints.size() << "���� " << endl; //������� ���
}

void cvHarrisCorner() {
    Mat figure[4]; //�ﰢ��,�簢��,������,������ �̹����� �о� figure�迭�� �ֱ�
    figure[0] = imread("triangle.png");
    figure[1] = imread("rect.png");
    figure[2] = imread("pentagon.png");
    figure[3] = imread("hexagon.png");

    for (int i = 0; i < 4; i++) { //4�������� ����������
        Mat img = figure[i]; 
        
        if (img.empty()) {
            cout << "Empty image!\n";
            exit(1);
        }

        resize(img, img, Size(500, 500), 0, 0, INTER_CUBIC); //500,500���� resize

        Mat gray;
        cvtColor(img, gray, CV_BGR2GRAY); //����̹����� ��ȯ

        Mat harr;
        cornerHarris(gray, harr, 2, 3, 0.04, BORDER_DEFAULT); //harriscorner
        normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat()); //����ȭ

        Mat harr_abs;
        convertScaleAbs(harr, harr_abs); //�������� ��ȯ

        //Print corners
        int thresh = 125;
        Mat result = img.clone();
        for (int y = 0; y < harr.rows; y += 1) {
            for (int x = 0; x < harr.cols; x += 1) {
                if ((int)harr.at<float>(y, x) > thresh)
                    circle(result, Point(x, y), 7, Scalar(255, 0, 255), 0, 4, 0); //�ڳʺκ��� ������ ǥ��
            }
        }
        cvBlobDetection(result); //blob detection ����
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
    params.minThreshold = 10; //�ּ� threshold��
    params.maxThreshold = 300; //�ִ� threshold�� 
    params.filterByArea = true; //�ּ� �ִ� ���� ����
    params.minArea = 300; //�ּ�
    params.maxArea = 10000; //�ִ�
    params.filterByCircularity = true; //������ �󸶳� ������� ����
    params.minCircularity = 0.65;
    params.filterByConvexity = true; //���ΰ� �󸶳� ������ ä�������� ����
    params.minConvexity = 0.9;
    params.filterByInertia = true; //Ÿ���� ���� �󸶳� ������� ����
    params.minInertiaRatio = 0.04;

    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params); //blob detector����

    std::vector<KeyPoint> keypoints;
    detector->detect(img, keypoints); //�̹������� keypoints ����

    Mat result;
    //result�̹����� ������ blob �׸���
    drawKeypoints(img, keypoints, result, Scalar(255, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("keypoints", result);

    cout << "������ ����: " << keypoints.size() << endl;// size()�� ���� ���� ���� ���
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
    warpPerspective(src, dst, pers_mat, Size(1200, 800)); //src_p�� dst_p�� ��ġ�� ������
    return dst;
}

void cvFeatureSIFT() {
    Mat img = imread("church.jpg", 1);
    // /*orginal sift*/
    Mat gray;
    cvtColor(img, gray, CV_BGR2GRAY); //grayscale�� ��ȯ
    Ptr<SiftFeatureDetector>detector = SiftFeatureDetector::create(); //detector����
    std::vector<KeyPoint> keypoints;
    detector->detect(gray, keypoints); //keypoints����

    Mat original;
    drawKeypoints(img, keypoints, original);
    imwrite("sift_original.jpg", original);
    imshow("Sift original", original); //original SIFT�̹��� ���

    /* ���ú�ȯ & ��⺯ȭ �� SIFT */
    img = warpPers(img); //���ú�ȯ ����
    cvtColor(img, gray, CV_BGR2GRAY); //grayscale��ȯ
    gray += 50; //��� 50 �ø���
    detector = SiftFeatureDetector::create();
    detector->detect(gray, keypoints);

    Mat result;
    drawKeypoints(img, keypoints, result);
    imwrite("sift_result.jpg", result);
    imshow("Sift result", result); //���ú�ȯ,��⺯ȯ �� SIFT�̹��� ���
    waitKey(0);
    destroyAllWindows();
}


int main() {
    //cvCoinDetection(); 
    //cvHarrisCorner();
    cvFeatureSIFT();
    return 0;
}
