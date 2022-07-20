#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace cv;



Mat getMyRotationMarix(Point center, double theta) {
    double a = cos(theta);
    double b = sin(theta);

    Mat matrix = (Mat_<double>(2, 3) << a, b, ((1 - a) * center.x - b * center.y),
        -b, a, (b * center.x + (1 - a) * center.y));
    return matrix;
}

void myRotation() {
    Mat src = imread("Lenna.png", 1);
    Mat dst, matrix;

    Point center = Point(src.cols / 2, src.rows / 2);
    matrix = getMyRotationMarix(center, 45.0);
    warpAffine(src, dst, matrix, src.size());

    imwrite("nonrot.jpg", src);
    imwrite("rot.jpg", dst);

    imshow("nonrot", src);
    imshow("rot", dst);
    waitKey(0);

    destroyAllWindows();
}

void cvRotation() {
    Mat src = imread("Lenna.png", 1);
    Mat dst, matrix;

    Point center = Point(src.cols / 2, src.rows / 2);
    matrix = getRotationMatrix2D(center, 45.0, 1.0);
    warpAffine(src, dst, matrix, src.size());

    imwrite("nonrot.jpg", src);
    imwrite("rot.jpg", dst);

    imshow("nonrot", src);
    imshow("rot", dst);
    waitKey(0);

    destroyAllWindows();
}


class Corner {
public:
    int x, y;
};

void myPerspective() {
    Mat src = imread("card_per.png", 1);
   
    Mat gray;
    cvtColor(src, gray, CV_BGR2GRAY); //����̹����� ��ȯ
    
   Mat harr;
    cornerHarris(gray, harr, 2, 3, 0.06, BORDER_DEFAULT); //HarrisCorner
    normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat()); //����ȭ
    Mat harr_abs;
    convertScaleAbs(harr, harr_abs); //�������� ��ȯ

    Corner corner[200]; //corner Ŭ���� ���
    Point2f srcQuad[4]; //�����̹����� ī�� ������

    int i = 0;
    //Print corners
    int thresh = 115;
    Mat result = src.clone();
    for (int y = 0; y < harr.rows; y += 1) {
        for (int x = 0; x < harr.cols; x += 1) {
            if ((int)harr.at<float>(y, x) > thresh) { //threshold������ �۴ٸ�
                if ((int)harr.at<float>(y, x) > (int)harr.at<float>(y - 1, x) && // ���� ��ǥ�� �������� �ڳ� ������ ����
                    (int)harr.at<float>(y, x) > (int)harr.at<float>(y + 1, x) &&
                    (int)harr.at<float>(y, x) > (int)harr.at<float>(y, x - 1) &&
                    (int)harr.at<float>(y, x) > (int)harr.at<float>(y, x + 1)) {
                    circle(result, Point(x, y), 5, Scalar(255, 0, 255), 0, 4, 0); //�ڳʺκ��� ������ ǥ��
                    corner[i].x = x; //��ǥ�� cornerŬ���� �迭�� �߰�
                    corner[i].y = y;
                    i++;
                }
            }
        }
    }
    imshow("circle", result); //�ڳʸ� ������ ǥ���� �̹��� ���
    
    int max_y = 0; int min_y = 0; 
    /*x��ǥ�� ���� �߾Ӱ����� ���� �� y�ּҰ� ���ϱ�-> srcQuad[0]*/
    for (int j = 0; j < i; j++) {
        if (j == 0) {
            min_y = corner[j].y;
        }
        if (min_y >= corner[j].y && harr.cols / 2 > corner[j].x) {
            min_y = corner[j].y;
            srcQuad[0] = Point2f(corner[j].x, corner[j].y);//y�� �ּ��� �� x,y��ǥ�� srcQuad[0]���� ����
        }
    }

    /*x��ǥ�� ���� �߾Ӱ����� ���� �� y�ִ밪 ���ϱ�-> srcQuad[1]*/
    for (int j = 0; j < i; j++) {
        if (j == 0) {
            max_y= corner[j].y;
        }
        if (max_y <= corner[j].y && harr.cols/2 > corner[j].x) {
            max_y = corner[j].y;
            srcQuad[1] =Point2f(corner[j].x, corner[j].y);//y�� �ִ��� �� x,y��ǥ�� srcQuad[0]���� ����
        }
    }
    
    /*x��ǥ�� ���� �߾Ӱ����� Ŭ �� y�ּҰ� ���ϱ�-> srcQuad[2]*/
    for (int j = i-1; j >=0; j--) { //j=0���� ���� x��ǥ�� ���� �߾Ӱ����� ���������� �����ϹǷ�
        if (j==i-1) {                 //i-1���� j--�ϸ鼭 ����
            min_y = corner[j].y;
        }
        if (min_y >= corner[j].y && harr.cols / 2 <= corner[j].x) {
            min_y = corner[j].y;
            srcQuad[2] = Point2f(corner[j].x, corner[j].y); //y�� �ּ��� �� x,y��ǥ�� srcQuad[2]���� ����
        }
    }
    /*x��ǥ�� ���� �߾Ӱ����� Ŭ �� y�ִ밪 ���ϱ�-> srcQuad[3]*/
    for (int j = 0; j < i; j++) {
        if (j == 0) {
            max_y = corner[j].y;
        }
        if (max_y <= corner[j].y && harr.cols / 2 <= corner[j].x) {
            max_y = corner[j].y;
            srcQuad[3] = Point2f(corner[j].x, corner[j].y); //y�� �ִ��� �� x,y��ǥ�� srcQuad[3]���� ����
        }
    }
    Mat dst, matrix;
    Point2f dstQuad[4];
    dstQuad[0] = Point2f(50, 140);
    dstQuad[1] = Point2f(50, 360);
    dstQuad[2] = Point2f(450, 140);
    dstQuad[3] = Point2f(450, 360);

    matrix = getPerspectiveTransform(srcQuad, dstQuad); //src�� dst�� ��ġ�� ������
    warpPerspective(src, dst, matrix, src.size());

    imwrite("nonper.jpg", src);
    imwrite("per.jpg", dst);

    imshow("nonper", src);
    imshow("per", dst);


    waitKey(0);
    destroyAllWindows();

}


int main() {
  //  myRotation();
//    cvRotation();
    myPerspective();

    return 0;
}

