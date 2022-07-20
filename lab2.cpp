#include <iostream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;
void SpreadSalts(Mat& , int ,int,int);
void SaltsCount(Mat&);
void darkup(Mat&);
void darkdown(Mat&);
Mat GetHistogram(Mat&);

int main() {
	//Mat src_img = imread("img1.jpg", 1); // 이미지 컬러로 읽기
	//Mat src_img = imread("img2.jpg", 0); // 이미지 흑백영상으로 읽기
	//SpreadSalts(src_img, 70,30,50); //점 흩뿌리기
	//SaltsCount(src_img);
	//darkup(src_img);
	//darkdown(src_img);
	//imshow("Test window", src_img); // 이미지 출력
	//imshow("histogram", GetHistogram(src_img));

	//-------------------------------------------------------------------------
	//주어진 영상(img3.jpg, img4.jpg, img5.jpg)을 이용해 다음의 영상을 완성할 것
	Mat imgA = imread("img3.jpg", 1);
	Mat imgB = imread("img4.jpg", 1);
	resize(imgB, imgB, Size(imgA.cols, imgA.rows)); //imgB의 크기를 imagA의 크기와 같도록 설정
	Mat dist1;
	subtract(imgA, imgB, dist1); //빼기 연산을 해서 dist1에 저장
	imshow("A+B",dist1);

	Mat srcrgb = imread("img5.jpg", 1); //컬러영상 읽어오기
	Mat gray_img, mask;
	cvtColor(srcrgb, gray_img, CV_BGR2GRAY); //흑백으로 만들어 gray_img에 저장
	Mat imageROI(dist1, Rect(gray_img.cols/2, 330, gray_img.cols, gray_img.rows)); //관심영역 설정
	threshold(gray_img, mask, 178, 255, THRESH_BINARY); 
	
    mask=120 - mask; // 그레이스케일에서 CV_8U(부호가없는 8비트)이기 때문에 127을 넘는 픽셀들은 모두 0이됨
	
	srcrgb.copyTo(imageROI, mask); //srcrgb영상을 관심영역에 mask를 적용하여 복사
	imshow("Spacex", dist1);

	waitKey(0); // 키 입력 대기(0: 키가 입력될 때 까지 프로그램 멈춤)
	destroyWindow("Test window"); // 이미지 출력창 종료
	destroyWindow("Spacex"); // 이미지 출력창 종료
	return 0;
}






void darkdown(Mat& img) { //밑으로 갈수록 어두워짐
	
	for (int x = 0; x < img.rows; x++) {
		for (int y = 0; y < img.cols; y++) {
			float g = float(x + 1) / float(img.rows); // x가 0이면 나눌 수 없기때문에 x+1로 나누어줌
			if (img.at<uchar>(x, y) - g*255 > 0) 
				 img.at<uchar>(x, y) -= g*255; 
			else
				img.at<uchar>(x, y) = 0;
		}
	}
	
}

void darkup(Mat& img) { //위로 갈수록 어두워짐

	for (int x = 0; x < img.rows; x++) {
		for (int y = 0; y < img.cols; y++) {
			float g = float(x + 1) / float(img.rows); // x가 0이면 나눌 수 없기때문에 x+1로 나누어줌
			if (img.at<uchar>(x, y) - (255 - g * 255) > 0)
				img.at<uchar>(x, y) -= (255- g * 255);
			else
				img.at<uchar>(x, y) = 0;
		}
	}
}


void SaltsCount(Mat& src) {

	int b = 0;
	int r = 0;
	int g = 0;
	
	for (int x = 0; x < src.rows; x++) {
		for (int y = 0; y < src.cols; y++) {
			if (src.at<Vec3b>(x, y) == Vec3b(255,0,0)) b++;
			else if (src.at<Vec3b>(x, y) == Vec3b(0, 255, 0)) g++;
			else if (src.at<Vec3b>(x, y) == Vec3b(0, 0, 255)) r++;
		}
	}
	cout <<"blue = "<< b <<"\nred = "<< r <<"\ngreen = " << g;
}


void SpreadSalts( Mat& img, int b, int g , int r) {
	
	for (int n = 0; n < b;  n++) {
		
			int x1 = rand() % img.cols;  
			int y1 = rand() % img.rows;
			/* 나머지는 나누는 수를 넘을 수 없으므로 위치를 제한시키기위한 moduler연산*/
			if (img.channels() == 3)  //이미지가 컬러일 때
				img.at<Vec3b>(y1, x1) = Vec3b(255, 0, 0); //컬러를 blue로
	}
	for (int n = 0; n < g; n++) {
		int x2 = rand() % img.cols;
		int y2 = rand() % img.rows;
		
		if (img.channels() == 3) 
			img.at<Vec3b>(y2, x2) = Vec3b(0, 255, 0); //컬러를 green으로
	}
	for (int n = 0; n < r; n++) {
		int x3 = rand() % img.cols;
		int y3 = rand() % img.rows;
		if (img.channels() == 3)
			img.at<Vec3b>(y3, x3) = Vec3b(0, 0, 255); //컬러를 red으로
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
