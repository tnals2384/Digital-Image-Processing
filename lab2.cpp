#include <iostream>
#include "opencv2/core/core.hpp" // Mat class�� ���� data structure �� ��� ��ƾ�� �����ϴ� ���
#include "opencv2/highgui/highgui.hpp" // GUI�� ���õ� ��Ҹ� �����ϴ� ���(imshow ��)
#include "opencv2/imgproc/imgproc.hpp" // ���� �̹��� ó�� �Լ��� �����ϴ� ���
using namespace cv;
using namespace std;
void SpreadSalts(Mat& , int ,int,int);
void SaltsCount(Mat&);
void darkup(Mat&);
void darkdown(Mat&);
Mat GetHistogram(Mat&);

int main() {
	//Mat src_img = imread("img1.jpg", 1); // �̹��� �÷��� �б�
	//Mat src_img = imread("img2.jpg", 0); // �̹��� ��鿵������ �б�
	//SpreadSalts(src_img, 70,30,50); //�� ��Ѹ���
	//SaltsCount(src_img);
	//darkup(src_img);
	//darkdown(src_img);
	//imshow("Test window", src_img); // �̹��� ���
	//imshow("histogram", GetHistogram(src_img));

	//-------------------------------------------------------------------------
	//�־��� ����(img3.jpg, img4.jpg, img5.jpg)�� �̿��� ������ ������ �ϼ��� ��
	Mat imgA = imread("img3.jpg", 1);
	Mat imgB = imread("img4.jpg", 1);
	resize(imgB, imgB, Size(imgA.cols, imgA.rows)); //imgB�� ũ�⸦ imagA�� ũ��� ������ ����
	Mat dist1;
	subtract(imgA, imgB, dist1); //���� ������ �ؼ� dist1�� ����
	imshow("A+B",dist1);

	Mat srcrgb = imread("img5.jpg", 1); //�÷����� �о����
	Mat gray_img, mask;
	cvtColor(srcrgb, gray_img, CV_BGR2GRAY); //������� ����� gray_img�� ����
	Mat imageROI(dist1, Rect(gray_img.cols/2, 330, gray_img.cols, gray_img.rows)); //���ɿ��� ����
	threshold(gray_img, mask, 178, 255, THRESH_BINARY); 
	
    mask=120 - mask; // �׷��̽����Ͽ��� CV_8U(��ȣ������ 8��Ʈ)�̱� ������ 127�� �Ѵ� �ȼ����� ��� 0�̵�
	
	srcrgb.copyTo(imageROI, mask); //srcrgb������ ���ɿ����� mask�� �����Ͽ� ����
	imshow("Spacex", dist1);

	waitKey(0); // Ű �Է� ���(0: Ű�� �Էµ� �� ���� ���α׷� ����)
	destroyWindow("Test window"); // �̹��� ���â ����
	destroyWindow("Spacex"); // �̹��� ���â ����
	return 0;
}






void darkdown(Mat& img) { //������ ������ ��ο���
	
	for (int x = 0; x < img.rows; x++) {
		for (int y = 0; y < img.cols; y++) {
			float g = float(x + 1) / float(img.rows); // x�� 0�̸� ���� �� ���⶧���� x+1�� ��������
			if (img.at<uchar>(x, y) - g*255 > 0) 
				 img.at<uchar>(x, y) -= g*255; 
			else
				img.at<uchar>(x, y) = 0;
		}
	}
	
}

void darkup(Mat& img) { //���� ������ ��ο���

	for (int x = 0; x < img.rows; x++) {
		for (int y = 0; y < img.cols; y++) {
			float g = float(x + 1) / float(img.rows); // x�� 0�̸� ���� �� ���⶧���� x+1�� ��������
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
			/* �������� ������ ���� ���� �� �����Ƿ� ��ġ�� ���ѽ�Ű������ moduler����*/
			if (img.channels() == 3)  //�̹����� �÷��� ��
				img.at<Vec3b>(y1, x1) = Vec3b(255, 0, 0); //�÷��� blue��
	}
	for (int n = 0; n < g; n++) {
		int x2 = rand() % img.cols;
		int y2 = rand() % img.rows;
		
		if (img.channels() == 3) 
			img.at<Vec3b>(y2, x2) = Vec3b(0, 255, 0); //�÷��� green����
	}
	for (int n = 0; n < r; n++) {
		int x3 = rand() % img.cols;
		int y3 = rand() % img.rows;
		if (img.channels() == 3)
			img.at<Vec3b>(y3, x3) = Vec3b(0, 0, 255); //�÷��� red����
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
