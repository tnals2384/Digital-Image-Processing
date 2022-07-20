#include <iostream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;


int myKernelConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height) {
	int sum = 0;
	int sumKernel = 0;

	for (int j = -1; j <= 1; j++) {
		for (int i = -1; i <= 1; i++) {
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
				sum += arr[(y + j) * width + (x + i)] * kernel[i + 1][j + 1];
				sumKernel += kernel[i + 1][j + 1];
			}
		}
	}

	if (sumKernel != 0) { return sum / sumKernel; }
	else return sum;
}

Mat myCopy(Mat srcImg) {
	int width = srcImg.cols;
	int height = srcImg.rows;


	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y * width + x] = srcData[y * width + x];
		}
	}

	return dstImg;
}

Mat myGaussianFilter(Mat srcImg) {
	int width = srcImg.cols;
	int height = srcImg.rows;
	int kernel[3][3] = { 1,2,1,
							2,4,2,
							1,2,1 };
	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y * width + x] = myKernelConv3x3(srcData, kernel, x, y, width, height);
		}
	}

	return dstImg;
}

Mat mySobelFilter(Mat srcImg) {
	int kernelX[3][3] = { -1, 0, 1,
								-2, 0, 2,
								-1, 0, 1 };
	int kernelY[3][3] = { -1, -2, -1,
								0, 0, 0,
								1, 2, 1 };
	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;
	int width = srcImg.cols;
	int height = srcImg.rows;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y * width + x] = (abs(myKernelConv3x3(srcData, kernelX, x, y, width, height)) +
				abs(myKernelConv3x3(srcData, kernelY, x, y, width, height))) / 2;
		}
	}
	return dstImg;
}

Mat mySampling(Mat srcImg) {
	int width = srcImg.cols / 2;
	int height = srcImg.rows / 2;
	Mat dstImg(height, width, CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y * width + x] = srcData[(y * 2) * (width * 2) + (x * 2)];
		}
	}
	return dstImg;
}

vector<Mat> myGaussianPyramid(Mat srcImg) {
	vector<Mat> Vec;
	Vec.push_back(srcImg);
	for (int i = 0; i < 4; i++) {
		srcImg = mySampling(srcImg);
		srcImg = myGaussianFilter(srcImg);

		Vec.push_back(srcImg);
	}
	return Vec;
}

vector<Mat> myLaplacianPyramid(Mat srcImg) {
	vector<Mat> Vec;

	for (int i = 0; i < 4; i++) {
		if (i != 3) {
			Mat highImg = srcImg;

			srcImg = mySampling(srcImg);
			srcImg = myGaussianFilter(srcImg);

			Mat lowImg = srcImg;
			resize(lowImg, lowImg, highImg.size());
			Vec.push_back(highImg - lowImg + 128);
		}
		else {
			Vec.push_back(srcImg);
		}
	}
	return Vec;
}

vector<Mat> myLaplacianPyramidReverse(Mat srcImg) {
	vector<Mat> VecLap = myLaplacianPyramid(srcImg);
	Mat dstImg;

	reverse(VecLap.begin(), VecLap.end());
	for (int i = 0; i < VecLap.size(); i++) {
		if (i == 0) {
			dstImg = VecLap[i];
		}
		else {
			resize(dstImg, dstImg, VecLap[i].size());
			dstImg = dstImg + VecLap[i] - 128;
		}
	}
}

int main() {
	Mat src_img = imread("gear.jpg", 0);
	imshow("Test window", src_img);
	//Mat img=mySobelFilter(src_img);
	//myGaussianFilter(src_img);
	//imshow("sobel", img);
	//vector<Mat> gp = myGaussianPyramid(src_img);
	vector<Mat> gp = myLaplacianPyramid(src_img);
	imshow("Test_gp", gp[1]);
	imshow("Test_gp2", gp[2]);
	imshow("Test_gp3", gp[3]);
	//imshow("Test_gp4", gp[4]);
	waitKey(0);
	destroyWindow("Test window");
	return 0;
}
