#include <iostream>
#include <iomanip>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

Mat padding(Mat img) {
	int dftRows = getOptimalDFTSize(img.rows);
	int dftCols = getOptimalDFTSize(img.cols);

	Mat padded;
	copyMakeBorder(img, padded, 0, dftRows - img.rows, 0, dftCols - img.cols, BORDER_CONSTANT, Scalar::all(0));

	return padded;
}

Mat doDft(Mat src_img) {
	Mat float_img;
	src_img.convertTo(float_img, CV_32F);

	Mat complex_img;
	dft(float_img, complex_img, DFT_COMPLEX_OUTPUT);

	return complex_img;
}

Mat getMagnitude(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes);
	
	Mat magImg;
	magnitude(planes[0], planes[1], magImg);
	magImg += Scalar::all(1);
	log(magImg, magImg);

	return magImg;
}

Mat myNormalize(Mat src) {
	Mat dst;
	src.copyTo(dst);
	normalize(dst, dst, 0, 255, NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1);

	return dst;
}

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

Mat SDSobelFilter(Mat srcImg) {
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



Mat getPhase(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes); //실수부 허수부 분리

	Mat phaImg;
	phase(planes[0], planes[1], phaImg); //phase 취득

	return phaImg;
}

Mat centralize(Mat complex) { //좌표계중앙이동
	Mat planes[2];
	split(complex, planes);
	int cx = planes[0].cols / 2;
	int cy = planes[1].rows / 2;

	Mat q0Re(planes[0], Rect(0, 0, cx, cy));
	Mat q1Re(planes[0], Rect(cx, 0, cx, cy));
	Mat q2Re(planes[0], Rect(0, cy, cx, cy));
	Mat q3Re(planes[0], Rect(cx, cy, cx, cy));

	Mat tmp;
	q0Re.copyTo(tmp);
	q3Re.copyTo(q0Re);
	tmp.copyTo(q3Re);
	q1Re.copyTo(tmp);
	q2Re.copyTo(q1Re);
	tmp.copyTo(q2Re);

	Mat q0Im(planes[1], Rect(0, 0, cx, cy));
	Mat q1Im(planes[1], Rect(cx, 0, cx, cy));
	Mat q2Im(planes[1], Rect(0, cy, cx, cy));
	Mat q3Im(planes[1], Rect(cx, cy, cx, cy));

	q0Im.copyTo(tmp);
	q3Im.copyTo(q0Im);
	tmp.copyTo(q3Im);
	q1Im.copyTo(tmp);
	q2Im.copyTo(q1Im);
	tmp.copyTo(q2Im);

	Mat centerComplex;
	merge(planes, 2, centerComplex);
	return centerComplex;
}

Mat setComplex(Mat mag_img, Mat pha_img) {
	exp(mag_img, mag_img);
	mag_img -= Scalar::all(1);

	Mat planes[2];
	polarToCart(mag_img, pha_img, planes[0], planes[1]);

	Mat complex_img;
	merge(planes, 2, complex_img);

	return complex_img;
}

Mat doIdft(Mat complex_img) {
	Mat idftcvt;
	idft(complex_img, idftcvt);

	Mat planes[2];
	split(idftcvt, planes);

	Mat dst_img;
	magnitude(planes[0], planes[1], dst_img);
	normalize(dst_img, dst_img, 255, 0, NORM_MINMAX);
	dst_img.convertTo(dst_img, CV_8UC1);

	return dst_img;
}

Mat doLPF(Mat src_img) {
	Mat pad_img = padding(src_img);
	Mat complex_img = doDft(pad_img);
	Mat center_complex_img = centralize(complex_img);
	Mat mag_img = getMagnitude(center_complex_img);
	Mat pha_img = getPhase(center_complex_img);

	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(mag_img, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(mag_img, mag_img, 0, 1, NORM_MINMAX);

	Mat mask_img = Mat::zeros(mag_img.size(), CV_32F);
	circle(mask_img, Point(mask_img.cols / 2, mask_img.rows / 2), 20, Scalar::all(1), -1, -1, 0);

	Mat mag_img2;
	multiply(mag_img, mask_img, mag_img2);

	normalize(mag_img2, mag_img2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complex_img2 = setComplex(mag_img2, pha_img);
	Mat dst_img = doIdft(complex_img2);

	return myNormalize(dst_img);
}

Mat doHPF(Mat src_img) {
	Mat pad_img = padding(src_img);
	Mat complex_img = doDft(pad_img);
	Mat center_complex_img = centralize(complex_img);
	Mat mag_img = getMagnitude(center_complex_img);
	Mat pha_img = getPhase(center_complex_img);

	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(mag_img, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(mag_img, mag_img, 0, 1, NORM_MINMAX);

	Mat mask_img = Mat::ones(mag_img.size(), CV_32F);
	circle(mask_img, Point(mask_img.cols / 2, mask_img.rows / 2), 45, Scalar::all(0), -1, -1, 0);

	Mat mag_img2;
	multiply(mag_img, mask_img, mag_img2);

	normalize(mag_img2, mag_img2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complex_img2 = setComplex(mag_img2, pha_img);
	Mat dst_img = doIdft(complex_img2);

	return myNormalize(dst_img);
}

Mat doBPF(Mat src_img) {
	Mat pad_img = padding(src_img);
	Mat complex_img = doDft(pad_img);
	Mat center_complex_img = centralize(complex_img);
	Mat mag_img = getMagnitude(center_complex_img);
	Mat pha_img = getPhase(center_complex_img);

	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(mag_img, &minVal, &maxVal, &minLoc, &maxLoc); 
	normalize(mag_img, mag_img, 0, 1, NORM_MINMAX); // 이미지 정규화

	Mat mask_img = Mat::zeros(mag_img.size(), CV_32F); //mask를  전부 0으로 초기화
	circle(mask_img, Point(mask_img.cols / 2, mask_img.rows / 2), 80, Scalar::all(1), -1, -1, 0); // 반지름 80인 Scalar 모두 1인 원
	circle(mask_img, Point(mask_img.cols / 2, mask_img.rows / 2), 20, Scalar::all(0), -1, -1, 0); //반지름 20인 검정 원
	
	Mat mag_img2;
	multiply(mag_img, mask_img, mag_img2); //마스크와 mag_img 곱해서 저장
	imshow("mag_img2", mag_img2); //mag_img2 출력

	normalize(mag_img2, mag_img2, (float)minVal, (float)maxVal, NORM_MINMAX); //정규화
	/*2D IDFT*/
	Mat complex_img2 = setComplex(mag_img2, pha_img); 
	Mat dst_img = doIdft(complex_img2);

	return myNormalize(dst_img);
}

Mat myFilter(Mat src_img) {
	Mat pad_img = padding(src_img);
	Mat complex_img = doDft(pad_img);
	Mat center_complex_img = centralize(complex_img);
	Mat mag_img = getMagnitude(center_complex_img);
	Mat pha_img = getPhase(center_complex_img);

	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(mag_img, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(mag_img, mag_img, 0, 1, NORM_MINMAX);

	Mat mask_img = Mat::ones(mag_img.size(), CV_32F);

	
	circle(mask_img, Point(mask_img.cols / 2, mask_img.rows / 2 ) , 20, Scalar::all(0), -1, -1, 0); // 반지름 15인 Scalar 0 원
	circle(mask_img, Point(mask_img.cols / 2, mask_img.rows / 2), 1, Scalar::all(1), -1, -1, 0); //반지름 2인 Scalar 1 원
	



	Mat mag_img2;
	multiply(mag_img, mask_img, mag_img2);
	imshow("mask", mag_img2);
	
	normalize(mag_img2, mag_img2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complex_img2 = setComplex(mag_img2, pha_img);
	Mat dst_img = doIdft(complex_img2);

	return myNormalize(dst_img);
}

void ex1() {
	Mat img = imread("img1.jpg", 0);
	imshow("original", img);
	Mat dstImg;
	dstImg = doBPF(img);
	imshow("Test window", dstImg);
	waitKey(0);
	destroyWindow("original");
	destroyWindow("Test window");
}

void ex2() {
	Mat img = imread("img2.jpg", 0);
	imshow("img", img);
	Mat SdstImg, FdstImg;
	SdstImg = SDSobelFilter(img);
	imshow("SpatialDomain", SdstImg);

	FdstImg = doHPF(img);
	imshow("FrequencyDomain", FdstImg);

	waitKey(0);
	destroyWindow("SpatialDomain");
	destroyWindow("FrequencyDomain");

	destroyWindow("img");

}

void ex3() {
	Mat img = imread("img3.jpg", 0);
	imshow("img", img);
	
	Mat dstImg;
	dstImg = myFilter(img);
	imshow("test",dstImg);
	waitKey(0);

}
int main() {

	//ex1();
	//ex2();
	ex3();
	return 0;
}

