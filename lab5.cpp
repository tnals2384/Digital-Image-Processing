#include <iostream>
#include <iomanip>
#include <algorithm>
#include "opencv2/core/core.hpp" // Mat class�� ���� data structure �� ��� ��ƾ�� �����ϴ� ���
#include "opencv2/highgui/highgui.hpp" // GUI�� ���õ� ��Ҹ� �����ϴ� ���(imshow ��)
#include "opencv2/imgproc/imgproc.hpp" // ���� �̹��� ó�� �Լ��� �����ϴ� ���
using namespace cv;
using namespace std;

double gaussian(float x, double sigma) {
	return exp(-(pow(x, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2)); //gaussian �Ŀ� ���� �� ��� �� return
}

float distance(int x, int y, int i, int j) {
	return float(sqrt(pow(x - i, 2) + pow(y - j, 2))); //�� �� ���� �Ÿ��� return
}


void myMedian(const Mat& src_img, Mat& dst_img, const Size& kn_size) {
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	int wd = src_img.cols; int hg = src_img.rows;   //�����̹����� width, height
	int kwd = kn_size.width; int khg = kn_size.height; //Ŀ���� width,height
	int rad_w = kwd / 2; int rad_h = khg / 2; 

	uchar* src_data = (uchar*)src_img.data;
	uchar* dst_data = (uchar*)dst_img.data;

	float* table = new float[kwd*khg](); //Ŀ�����̺� �����Ҵ�
	float tmp;
	
	//�����ڸ� ���� �ȼ��ε���
	for (int c = rad_w + 1; c < wd - rad_w; c++) {
		for (int r = rad_h + 1; r < hg - rad_h; r++) {
			tmp = 0.f;
			//Ŀ���ε���
			for (int kc = -rad_w; kc <= rad_w; kc++) {
				for (int kr = -rad_h; kr <= rad_h; kr++) {
					tmp = (float)src_data[(r + kr) * wd + (c + kc)]; //�����̹����� �����͸� tmp�� ����
					table[(kr + rad_h) * kwd + (kc + rad_w)] = tmp; //���̺� tmp ����
					
				}
			}
			sort(table, table+(kwd*khg)); //table�� ���� (alogrithm�� ���Ե� sort�Լ�)
			float med = table[kwd * khg / 2]; //�߰��� ã��
			dst_data[r * wd + c] = (uchar)med; //Ŀ�� �߰����� ����
		}
	}
	
	delete table;
}




void doMedianEx() {
	cout << "--- doMedianEx() --- \n" << endl;
	//�Է�
	Mat src_img = imread("salt_pepper.png", 0);
	if (!src_img.data) printf("No image data  \n");

	//<Median ���͸� ����>
	Mat dst_img;
	myMedian(src_img, dst_img, Size(3, 3));

	//<���>
	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("doMedianEx()", result_img);
	waitKey(0);
}

void bilateral(const Mat& src_img, Mat& dst_img, int c, int r, int diameter, double sig_r, double sig_s) {
	int radius = diameter / 2;
	double gr, gs, wei;
	double tmp = 0.;
	double sum = 0.;
	//<Ŀ�� �ε���>
	for (int kc = -radius; kc <= radius; kc++) {
		for (int kr = -radius; kr <= radius; kr++) {
			gr = gaussian((float)src_img.at<uchar>(c + kc, r + kr) - (float)src_img.at<uchar>(c, r),  sig_r);
			//range calc
			gs = gaussian(distance(c, r, c + kc, r + kr), sig_s);
			//spatial calc
			wei = gr * gs;
			tmp += src_img.at<uchar>(c + kc, r + kr) * wei;
			sum += wei;
		}
	}
	dst_img.at<double>(c, r) = tmp / sum; //����ȭ
}

void myBilateral(const Mat& src_img, Mat& dst_img, int diameter, double sig_r, double sig_s) {
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	Mat guide_img = Mat::zeros(src_img.size(), CV_64F);
	int wh = src_img.cols; int hg = src_img.rows;
	int radius = diameter / 2;
	//<�ȼ� �ε��� (�����ڸ� ����)>
	for (int c = radius + 1; c < hg - radius; c++) {
		for (int r = radius + 1; r < wh - radius; r++) {
			bilateral(src_img, guide_img, c, r, diameter, sig_r, sig_s);
			//ȭ�Һ� bilateral ���
		}
	}
	guide_img.convertTo(dst_img, CV_8UC1);
}


void doBilateralExs2() {
	cout << "--- doBilateralEx() sig_s=2 ---  \n" << endl;
	// < �Է� > 
	Mat src_img = imread("rock.png", 0);
	Mat dst_img[3];
	if (!src_img.data) printf("No image data \n");
	// < Bilateral ���͸� ���� >
	myBilateral(src_img, dst_img[0], 4, 20, 2);
	myBilateral(src_img, dst_img[1], 4, 100, 2);
	myBilateral(src_img, dst_img[2], 4, 999, 2);
	
	hconcat(dst_img[0], dst_img[1], dst_img[1]);
	hconcat(dst_img[1], dst_img[2], dst_img[2]); //�̹��� ��ġ��
 
	// < ���>
	imshow("sigma_s=2", dst_img[2]);
	waitKey(0);
}
void doBilateralExs6() {
	cout << "--- doBilateralEx() sig_s=6 ---  \n" << endl;
	// < �Է� >  
	Mat src_img = imread("rock.png", 0);
	Mat dst_img[3];
	if (!src_img.data) printf("No image data \n");
	// < Bilateral ���͸� ���� >
	myBilateral(src_img, dst_img[0], 12, 20, 6);
	myBilateral(src_img, dst_img[1], 12, 100, 6);
	myBilateral(src_img, dst_img[2], 12, 999, 6);

	hconcat(dst_img[0], dst_img[1], dst_img[1]);
	hconcat(dst_img[1], dst_img[2], dst_img[2]);


	// < ���>
	imshow("sigma_s=6", dst_img[2]);
	waitKey(0);
}
void doBilateralExs18() {
	cout << "--- doBilateralEx() sig_s=18 ---  \n" << endl;
	// < �Է� >  
	Mat src_img = imread("rock.png", 0);
	Mat dst_img[3];
	if (!src_img.data) printf("No image data \n");
	// < Bilateral ���͸� ���� >
	myBilateral(src_img, dst_img[0], 36, 20, 18.0);
	myBilateral(src_img, dst_img[1], 36, 100, 18.0);
	myBilateral(src_img, dst_img[2], 36, 999, 18.0);


	hconcat(dst_img[0], dst_img[1], dst_img[1]);
	hconcat(dst_img[1], dst_img[2], dst_img[2]);


	// < ���>
	imshow("sigma_s=18", dst_img[2]);
	waitKey(0);
}

void doCannyEx1() {
	cout << "--- doCannyEx()1 ---  \n" << endl;

	clock_t start, end;
	double result;
	
	start = clock(); //�ð� ����
	Mat src_img = imread("edge_test.jpg", 0);
	if (!src_img.data) printf("No image data \n");
	Mat dst_img;
	Canny(src_img, dst_img, 220, 240);
	end = clock(); //��
	result = (double)(end - start);
	cout << "result : " << ((result) / CLOCKS_PER_SEC) << "seconds" << endl;
	
	hconcat(src_img, dst_img, dst_img);
	imshow("doCannyEx1()", dst_img);
		
	waitKey(0);
}
void doCannyEx2() {
	cout << "--- doCannyEx()2 ---  \n" << endl;

	clock_t start, end;
	double result;

	start = clock(); //�ð� ����
	Mat src_img = imread("edge_test.jpg", 0);
	if (!src_img.data) printf("No image data \n");
	Mat dst_img;
	Canny(src_img, dst_img, 100, 240);
	end = clock(); //��
	result = (double)(end - start);

	cout << "result : " << ((result) / CLOCKS_PER_SEC) << "seconds" << endl;

	hconcat(src_img, dst_img, dst_img);
	imshow("doCannyEx()2", dst_img);

	waitKey(0);
}
void doCannyEx3() {
	cout << "--- doCannyEx3() ---  \n" << endl;

	clock_t start, end;
	double result;

	start = clock(); //�ð� ����
	Mat src_img = imread("edge_test.jpg", 0);
	if (!src_img.data) printf("No image data \n");
	Mat dst_img;
	Canny(src_img, dst_img, 30, 240);
	end = clock(); //��
	result = (double)(end - start);

	cout << "result : " << ((result) / CLOCKS_PER_SEC) << "seconds" << endl;

	hconcat(src_img, dst_img, dst_img);
	imshow("doCannyEx3()", dst_img);

	waitKey(0);
}

int main()  {
	//ex1
	//doMedianEx();

	//ex2
	//doCannyEx1();
	//doCannyEx2();
	//doCannyEx3();
	//ex3
	doBilateralExs2();
	doBilateralExs6();
	doBilateralExs18();

	return 0;
}