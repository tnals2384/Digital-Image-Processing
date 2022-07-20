#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string.h>
#include "opencv2/core/core.hpp" // Mat class�� ���� data structure �� ��� ��ƾ�� �����ϴ� ���
#include "opencv2/highgui/highgui.hpp" // GUI�� ���õ� ��Ҹ� �����ϴ� ���(imshow ��)
#include "opencv2/imgproc/imgproc.hpp" // ���� �̹��� ó�� �Լ��� �����ϴ� ���
using namespace cv;
using namespace std;


void createClustersInfo(Mat imgInput, int n_cluster, vector<Scalar>& clustersCenters,
	vector<vector<Point>>& ptInClusters) { 

	RNG random(cv::getTickCount()); //�������� �����ϴ� �Լ�

	for (int k = 0; k < n_cluster; k++) {
		//������ ��ǥ ȹ��
		Point centerKPoint;
		centerKPoint.x = random.uniform(0, imgInput.cols);
		centerKPoint.y = random.uniform(0, imgInput.rows);
		Scalar centerPixel = imgInput.at<Vec3b>(centerKPoint.y, centerKPoint.x);
		//������ ��ǥ�� ȭ�Ұ����� ������ �߾Ӱ��� ������
		Scalar centerK(centerPixel.val[0], centerPixel.val[1], centerPixel.val[2]);
		clustersCenters.push_back(centerK);

		vector<Point>ptInClustersK;
		ptInClusters.push_back(ptInClustersK);
	}
}


double computeColorDistance(Scalar pixel, Scalar clusterPixel) { //�Ÿ��� ���

	double diffBlue = pixel.val[0] - clusterPixel[0];
	double diffGreen = pixel.val[1] - clusterPixel[1];
	double diffRed = pixel.val[2] - clusterPixel[2];

	double distance = sqrt(pow(diffBlue, 2) + pow(diffGreen, 2) + pow(diffRed, 2));  //��Ŭ���� �Ÿ�

	return distance;
}

void findAssociatedCluster(Mat imgInput, int n_cluster,
	vector<Scalar> clustersCenters, vector<vector<Point>>& ptInClusters) {
	for (int r = 0; r < imgInput.rows; r++) {
		for (int c = 0; c < imgInput.cols; c++) {
			double minDistance = INFINITY;
			int closestClusterIndex = 0;
			Scalar pixel = imgInput.at<Vec3b>(r, c);

			for (int k = 0; k < n_cluster; k++) { //������ ���
				//�� ���� �߾Ӱ����� ���� ���
				Scalar clusterPixel = clustersCenters[k];
				double distance = computeColorDistance(pixel, clusterPixel);

				//���̰� ���� ���� �������� ��ǥ�� ���� �Ǻ�
				if (distance < minDistance) {
					minDistance = distance;
					closestClusterIndex = k;
				}
			}
			//��ǥ ����
			ptInClusters[closestClusterIndex].push_back(Point(c, r));
		}
	}
}

double adjustClusterCenters(Mat src_img, int n_cluster,
	vector<Scalar> clustersCenters, vector<vector<Point>>& ptInClusters,
		double& oldCenter, double newCenter) {
	
	double diffChange;

	for (int k = 0; k < n_cluster; k++) { //���������
		vector<Point> ptInCluster = ptInClusters[k];
		double newBlue = 0;
		double newGreen = 0;
		double newRed = 0;

		//��հ����
		for (int i = 0; i < ptInCluster.size(); i++) {
			Scalar pixel = src_img.at<Vec3b>(ptInCluster[i].y, ptInCluster[i].x);
			newBlue += pixel.val[0];
			newGreen += pixel.val[1];
			newRed += pixel.val[2];
		}
		newBlue /= ptInCluster.size();
		newGreen /= ptInCluster.size();
		newRed /= ptInCluster.size();

		//����� ��հ����� ���� �߾Ӱ� ��ü
		Scalar newPixel(newBlue, newGreen, newRed);
		newCenter += computeColorDistance(newPixel, clustersCenters[k]);
		//��� ������ ���� ��հ��� ���� ���
		clustersCenters[k] = newPixel;
	}
	newCenter /= n_cluster;
	diffChange = abs(oldCenter - newCenter);
	//��� ������ ���� ��հ� ��ȭ�� ���

	oldCenter = newCenter;

	return diffChange;
}

Mat applyFinalClusterTolmage(Mat src_img, int n_cluster,
	 vector<vector<Point>>& ptInClusters, vector<Scalar> clustersCenters) {

	Mat dst_img(src_img.size(), src_img.type());

	for (int k = 0; k < n_cluster; k++) { //��� ����������
		vector<Point> ptInCluster = ptInClusters[k]; //������ ��ǥ��
		//���� color �����
		clustersCenters[k].val[0] = rand() % 255;
		clustersCenters[k].val[1] = rand() % 255;
		clustersCenters[k].val[2] = rand() % 255;
		for (int j = 0; j < ptInCluster.size(); j++) {
			//������ ��ǥ ��ġ�� �ִ� ȭ�� ���� �ش� ���� ���������� ��ü
			dst_img.at<Vec3b>(ptInCluster[j])[0] = clustersCenters[k].val[0];
			dst_img.at<Vec3b>(ptInCluster[j])[1] = clustersCenters[k].val[1];
			dst_img.at<Vec3b>(ptInCluster[j])[2] = clustersCenters[k].val[2];
		}
	}
	return dst_img;

}

Mat MyKmeans(Mat src_img, int n_cluster) {
	vector<Scalar>clustersCenters; //���� �߾Ӱ� ����
	vector<vector<Point>>ptInClusters; //������ ��ǥ ����
	double threshold = 0.001;
	double oldCenter = INFINITY;
	double newCenter = 0;
	double diffChange = oldCenter - newCenter; //���� ������ ��ȭ��

	// <�ʱ⼳��>
	//���� �߾Ӱ��� �������� �Ҵ� �� ������ ��ǥ���� ������ ���� �Ҵ�
	createClustersInfo(src_img, n_cluster, clustersCenters, ptInClusters);

	//<�߾Ӱ� ���� �� ȭ�Һ� ���� �Ǻ�>
	//�ݺ����� ������� ���� �߾Ӱ� ����
	//������ �Ӱ谪 ���� ���� ������ ��ȭ�� ���� ������ �ݺ�
	while (diffChange > threshold) {
     	//<�ʱ�ȭ>
		newCenter = 0;
		for (int k = 0; k < n_cluster; k++) { ptInClusters[k].clear(); }
		//<������ ���� �߾Ӱ��� �������� ���� Ž��>
		findAssociatedCluster(src_img, n_cluster, clustersCenters, ptInClusters);
		//<���� �߾Ӱ� ����>
		diffChange = adjustClusterCenters(src_img, n_cluster, clustersCenters, ptInClusters, oldCenter, newCenter);
	 }
	//<���� �߾Ӱ����θ� �̷���� ���� ����>
	Mat dst_img = applyFinalClusterTolmage(src_img, n_cluster, ptInClusters, clustersCenters);

	imshow("results", dst_img);
	//waitKey(0);


	return dst_img;
}

Mat MyBgr2Hsv(Mat src_img) {
	double b, g, r, h, s, v;
	h = 0; s = 0; v = 0;
	Mat dst_img(src_img.size(), src_img.type());

	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) { //��� �ȼ��� ����
			b = (double)src_img.at<Vec3b>(y, x)[0]/255.0; //������ 0~255 �̱� ������ 255�� ��������
			g = (double)src_img.at<Vec3b>(y, x)[1]/255.0;
			r = (double)src_img.at<Vec3b>(y, x)[2]/255.0;

			double max, min;
			//max(R,G,B) ã��
			max = (b < g) ? g : b; 
			max = (max > r) ? max : r;
			//min(R,G,B) ã��
			min = (b < g) ? b : g;
			min = (min < r)? min : r;

			//value���� max
			v = max;

			//saturation�� ���
			if (v > 0) {
				s= (max - min) / max;
			}
			else {
				s = 0;
				h = 0;
			}

			//hue�� ���. 0~360 ����
			if (h < 0) { h += 360; }
			else if (r >= max && (max - min) != 0) {
				h = (g - b) / (max - min) ;
				h *= 60.0;
			}
			else if (g >= max && (max - min) != 0) {
				h = 2 + ((b - r) / (max - min));
				h *= 60.0;
			}
			else if (b >= max && (max - min) != 0) {
				h = 4 +( (r - g) / (max - min) );
				h *= 60.0;
			}
			h /= 360;
			
			
			//dst_img�� 255�� �ٽ� ���ؼ� ���� h,s,v�� �Ҵ�. 
			dst_img.at<Vec3b>(y, x)[0] = (uchar)(h*255.0); 
			dst_img.at<Vec3b>(y, x)[1] = (uchar)(s*255.0);
			dst_img.at<Vec3b>(y, x)[2] = (uchar)(v*255.0);
		}
	}
	return dst_img;
}

Mat MyinRange(Mat src_img, Scalar lower, Scalar upper) {
	Mat dst_img = Mat::zeros(src_img.size(), src_img.type()); //�����̹����� ���� ������, ���� Ÿ�� ��� �̹��� ����
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) { //��� pixel�� ����
			Scalar pixel = src_img.at<Vec3b>(y, x);
			//�ȼ� ���� lower���� upper�� ������ ���̶��
			if (pixel.val[0] > lower.val[0] && pixel.val[1] > lower.val[1] &&
				pixel.val[2] > lower.val[2] && pixel.val[0] < upper.val[0] &&
				pixel.val[1] < upper.val[1] && pixel.val[2] < upper.val[2]) {
				dst_img.at<Vec3b>(y, x) = 255; //white�� ����
			}
		}
	}
	return dst_img;
}

void Extraction(Mat src_img) {
	Mat range_img;

	//green, blue, orange�� ���Ѱ� ���Ѱ� ����
	Scalar greenl(40, 60, 0);
	Scalar greenu(120, 255, 255);
	Scalar bluel(120, 50, 10 );
	Scalar blueu(190, 255, 255);
	Scalar orangel(0, 80, 150);
	Scalar orangeu(40, 360, 255);

	Mat hsv_img(src_img.size(), src_img.type()); 

	int orange = 0; int green = 0; int  blue = 0;
	hsv_img = MyBgr2Hsv(src_img); //hsv���� �����
	imshow("fruitbgr", src_img);
	imshow("fruithsv", hsv_img);

	for (int y = 0; y < hsv_img.rows; y++) {
		for (int x = 0; x < hsv_img.cols; x++) { //��� �ȼ�������
			if (hsv_img.at<Vec3b>(y, x)[0] >= 40 && hsv_img.at<Vec3b>(y, x)[0] <= 120)
				green++; // h���� 40~120 ���� ���̶�� green count
			else if (hsv_img.at<Vec3b>(y, x)[0] >= 0 && hsv_img.at<Vec3b>(y, x)[0] <= 40)
				orange++; //h���� 0~40 ���� ���̶��  orange count
			else if (hsv_img.at<Vec3b>(y, x)[0] >= 110 && hsv_img.at<Vec3b>(y, x)[0] <= 190)
				blue++; //h���� 110~190 ���� ���̶�� blue count 
		}
	}

 //count ���� ���帹�� color�� ���� inrange �Լ� �����ϰ� ���� ���
	if (green >= orange && green >= blue) {
		range_img = MyinRange(hsv_img, greenl, greenu);
		cout << "green" << endl;
	}
	else if (orange >= green && orange >= blue) {
		range_img = MyinRange(hsv_img, orangel, orangeu);
		cout << "orange" << endl;
	}
	else {
		range_img = MyinRange(hsv_img, bluel, blueu);
		cout << "blue" << endl;
	}

 
	for (int y = 0; y < hsv_img.rows; y++) {
		for (int x = 0; x < hsv_img.cols; x++) {
			if (range_img.at<Vec3b>(y, x) == Vec3b(0, 0, 0)) { //inrange ������ �̹����� pixel�� Scalar(0,0,0)�̸�
				src_img.at<Vec3b>(y, x) = Vec3b(0, 0, 0); //�����̹����� (0,0,0)���� ����
			}
		}
	}
	
	imshow("fruit", src_img); //���

	waitKey(0);

}

void ex1_orange() {
	Mat src_img = imread("orange.jpg", 1);
	Extraction(src_img);
}

void ex1_cucumber() {
	Mat src_img = imread("cucumber.png", 1);
	Extraction(src_img);

}

void ex1_blueberry() {
	Mat src_img = imread("blueberry.jpg", 1);
	Extraction(src_img);

}

void ex2() {
	Mat src_img = imread("fruit.jpg", 1);
	imshow("original", src_img);
	MyKmeans(src_img, 8);
	waitKey(0);
}
int main() {
//	ex1_orange();
//	ex1_blueberry();
//	ex1_cucumber();
	ex2();
	return 0;
}