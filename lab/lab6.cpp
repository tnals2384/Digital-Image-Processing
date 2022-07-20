#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string.h>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;


void createClustersInfo(Mat imgInput, int n_cluster, vector<Scalar>& clustersCenters,
	vector<vector<Point>>& ptInClusters) { 

	RNG random(cv::getTickCount()); //무작위값 설정하는 함수

	for (int k = 0; k < n_cluster; k++) {
		//무작위 좌표 획득
		Point centerKPoint;
		centerKPoint.x = random.uniform(0, imgInput.cols);
		centerKPoint.y = random.uniform(0, imgInput.rows);
		Scalar centerPixel = imgInput.at<Vec3b>(centerKPoint.y, centerKPoint.x);
		//무작위 좌표의 화소값으로 군집별 중앙값을 설정함
		Scalar centerK(centerPixel.val[0], centerPixel.val[1], centerPixel.val[2]);
		clustersCenters.push_back(centerK);

		vector<Point>ptInClustersK;
		ptInClusters.push_back(ptInClustersK);
	}
}


double computeColorDistance(Scalar pixel, Scalar clusterPixel) { //거리를 계산

	double diffBlue = pixel.val[0] - clusterPixel[0];
	double diffGreen = pixel.val[1] - clusterPixel[1];
	double diffRed = pixel.val[2] - clusterPixel[2];

	double distance = sqrt(pow(diffBlue, 2) + pow(diffGreen, 2) + pow(diffRed, 2));  //유클리드 거리

	return distance;
}

void findAssociatedCluster(Mat imgInput, int n_cluster,
	vector<Scalar> clustersCenters, vector<vector<Point>>& ptInClusters) {
	for (int r = 0; r < imgInput.rows; r++) {
		for (int c = 0; c < imgInput.cols; c++) {
			double minDistance = INFINITY;
			int closestClusterIndex = 0;
			Scalar pixel = imgInput.at<Vec3b>(r, c);

			for (int k = 0; k < n_cluster; k++) { //군집별 계산
				//각 군집 중앙값과의 차이 계산
				Scalar clusterPixel = clustersCenters[k];
				double distance = computeColorDistance(pixel, clusterPixel);

				//차이가 가장 적은 군집으로 좌표의 군집 판별
				if (distance < minDistance) {
					minDistance = distance;
					closestClusterIndex = k;
				}
			}
			//좌표 저장
			ptInClusters[closestClusterIndex].push_back(Point(c, r));
		}
	}
}

double adjustClusterCenters(Mat src_img, int n_cluster,
	vector<Scalar> clustersCenters, vector<vector<Point>>& ptInClusters,
		double& oldCenter, double newCenter) {
	
	double diffChange;

	for (int k = 0; k < n_cluster; k++) { //군집별계산
		vector<Point> ptInCluster = ptInClusters[k];
		double newBlue = 0;
		double newGreen = 0;
		double newRed = 0;

		//평균값계산
		for (int i = 0; i < ptInCluster.size(); i++) {
			Scalar pixel = src_img.at<Vec3b>(ptInCluster[i].y, ptInCluster[i].x);
			newBlue += pixel.val[0];
			newGreen += pixel.val[1];
			newRed += pixel.val[2];
		}
		newBlue /= ptInCluster.size();
		newGreen /= ptInCluster.size();
		newRed /= ptInCluster.size();

		//계산한 평균값으로 군집 중앙값 대체
		Scalar newPixel(newBlue, newGreen, newRed);
		newCenter += computeColorDistance(newPixel, clustersCenters[k]);
		//모든 군집에 대한 평균값도 같이 계산
		clustersCenters[k] = newPixel;
	}
	newCenter /= n_cluster;
	diffChange = abs(oldCenter - newCenter);
	//모든 군집에 대한 평균값 변화량 계산

	oldCenter = newCenter;

	return diffChange;
}

Mat applyFinalClusterTolmage(Mat src_img, int n_cluster,
	 vector<vector<Point>>& ptInClusters, vector<Scalar> clustersCenters) {

	Mat dst_img(src_img.size(), src_img.type());

	for (int k = 0; k < n_cluster; k++) { //모든 군집에대해
		vector<Point> ptInCluster = ptInClusters[k]; //군집별 좌표들
		//랜덤 color 만들기
		clustersCenters[k].val[0] = rand() % 255;
		clustersCenters[k].val[1] = rand() % 255;
		clustersCenters[k].val[2] = rand() % 255;
		for (int j = 0; j < ptInCluster.size(); j++) {
			//군집별 좌표 위치에 있는 화소 값을 해당 군집 랜덤값으로 대체
			dst_img.at<Vec3b>(ptInCluster[j])[0] = clustersCenters[k].val[0];
			dst_img.at<Vec3b>(ptInCluster[j])[1] = clustersCenters[k].val[1];
			dst_img.at<Vec3b>(ptInCluster[j])[2] = clustersCenters[k].val[2];
		}
	}
	return dst_img;

}

Mat MyKmeans(Mat src_img, int n_cluster) {
	vector<Scalar>clustersCenters; //군집 중앙값 벡터
	vector<vector<Point>>ptInClusters; //군집별 좌표 벡터
	double threshold = 0.001;
	double oldCenter = INFINITY;
	double newCenter = 0;
	double diffChange = oldCenter - newCenter; //군집 조정의 변화량

	// <초기설정>
	//군집 중앙값을 무작위로 할당 및 군집별 좌표값을 저장할 벡터 할당
	createClustersInfo(src_img, n_cluster, clustersCenters, ptInClusters);

	//<중앙값 조정 및 화소별 군집 판별>
	//반복적인 방법으로 군집 중앙값 조정
	//설정한 임계값 보다 군집 조정의 변화가 작을 때까지 반복
	while (diffChange > threshold) {
     	//<초기화>
		newCenter = 0;
		for (int k = 0; k < n_cluster; k++) { ptInClusters[k].clear(); }
		//<현재의 군집 중앙값을 기준으로 군집 탐색>
		findAssociatedCluster(src_img, n_cluster, clustersCenters, ptInClusters);
		//<군집 중앙값 조절>
		diffChange = adjustClusterCenters(src_img, n_cluster, clustersCenters, ptInClusters, oldCenter, newCenter);
	 }
	//<군집 중앙값으로만 이루어진 영상 생성>
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
		for (int x = 0; x < src_img.cols; x++) { //모든 픽셀에 대해
			b = (double)src_img.at<Vec3b>(y, x)[0]/255.0; //범위가 0~255 이기 때문에 255로 나누어줌
			g = (double)src_img.at<Vec3b>(y, x)[1]/255.0;
			r = (double)src_img.at<Vec3b>(y, x)[2]/255.0;

			double max, min;
			//max(R,G,B) 찾기
			max = (b < g) ? g : b; 
			max = (max > r) ? max : r;
			//min(R,G,B) 찾기
			min = (b < g) ? b : g;
			min = (min < r)? min : r;

			//value값은 max
			v = max;

			//saturation값 계산
			if (v > 0) {
				s= (max - min) / max;
			}
			else {
				s = 0;
				h = 0;
			}

			//hue값 계산. 0~360 범위
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
			
			
			//dst_img에 255를 다시 곱해서 구한 h,s,v값 할당. 
			dst_img.at<Vec3b>(y, x)[0] = (uchar)(h*255.0); 
			dst_img.at<Vec3b>(y, x)[1] = (uchar)(s*255.0);
			dst_img.at<Vec3b>(y, x)[2] = (uchar)(v*255.0);
		}
	}
	return dst_img;
}

Mat MyinRange(Mat src_img, Scalar lower, Scalar upper) {
	Mat dst_img = Mat::zeros(src_img.size(), src_img.type()); //원본이미지와 같은 사이즈, 같은 타입 흑백 이미지 생성
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) { //모든 pixel에 대해
			Scalar pixel = src_img.at<Vec3b>(y, x);
			//픽셀 값이 lower값과 upper값 사이의 값이라면
			if (pixel.val[0] > lower.val[0] && pixel.val[1] > lower.val[1] &&
				pixel.val[2] > lower.val[2] && pixel.val[0] < upper.val[0] &&
				pixel.val[1] < upper.val[1] && pixel.val[2] < upper.val[2]) {
				dst_img.at<Vec3b>(y, x) = 255; //white로 설정
			}
		}
	}
	return dst_img;
}

void Extraction(Mat src_img) {
	Mat range_img;

	//green, blue, orange의 하한값 상한값 설정
	Scalar greenl(40, 60, 0);
	Scalar greenu(120, 255, 255);
	Scalar bluel(120, 50, 10 );
	Scalar blueu(190, 255, 255);
	Scalar orangel(0, 80, 150);
	Scalar orangeu(40, 360, 255);

	Mat hsv_img(src_img.size(), src_img.type()); 

	int orange = 0; int green = 0; int  blue = 0;
	hsv_img = MyBgr2Hsv(src_img); //hsv영상 만들기
	imshow("fruitbgr", src_img);
	imshow("fruithsv", hsv_img);

	for (int y = 0; y < hsv_img.rows; y++) {
		for (int x = 0; x < hsv_img.cols; x++) { //모든 픽셀에대해
			if (hsv_img.at<Vec3b>(y, x)[0] >= 40 && hsv_img.at<Vec3b>(y, x)[0] <= 120)
				green++; // h값이 40~120 사이 값이라면 green count
			else if (hsv_img.at<Vec3b>(y, x)[0] >= 0 && hsv_img.at<Vec3b>(y, x)[0] <= 40)
				orange++; //h값이 0~40 사이 값이라면  orange count
			else if (hsv_img.at<Vec3b>(y, x)[0] >= 110 && hsv_img.at<Vec3b>(y, x)[0] <= 190)
				blue++; //h값이 110~190 사이 값이라면 blue count 
		}
	}

 //count 값이 가장많은 color에 따라 inrange 함수 적용하고 문자 출력
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
			if (range_img.at<Vec3b>(y, x) == Vec3b(0, 0, 0)) { //inrange 적용한 이미지의 pixel이 Scalar(0,0,0)이면
				src_img.at<Vec3b>(y, x) = Vec3b(0, 0, 0); //원본이미지도 (0,0,0)으로 설정
			}
		}
	}
	
	imshow("fruit", src_img); //출력

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
