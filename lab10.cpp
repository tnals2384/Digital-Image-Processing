#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/stitching.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

void ex_panorama_simple() {
	Mat img;
	vector<Mat> imgs;
	//left,center,right를 읽어와 imgs vector에 저장
	img = imread("left.jpg", IMREAD_COLOR);
	imgs.push_back(img);
	img = imread("center.jpg", IMREAD_COLOR);
	imgs.push_back(img);
	img = imread("right.jpg", IMREAD_COLOR);
	imgs.push_back(img);

	//stitcher class를 이용하여 imgs stitch
	Mat result;
	Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA, false);
	Stitcher::Status status = stitcher->stitch(imgs, result);
	if (status != Stitcher::OK) {
		cout << "Can't stitch images, error code = " << int(status) << endl;
		exit(-1);
	}

	imshow("ex_panorama_simple_result", result);
	imwrite("ex_panorama_simple_result.png", result);
	waitKey();
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


Mat makePanorama(Mat img_l, Mat img_r, int thresh_dist, int min_matches) {
	//grayscale로 변환
	Mat img_gray_l, img_gray_r;
	cvtColor(img_l, img_gray_l, CV_BGR2GRAY);
	cvtColor(img_r, img_gray_r, CV_BGR2GRAY);

	//SURF 방식으로 특징점 추출
	Ptr<SurfFeatureDetector> Detector = SURF::create(300);
	vector<KeyPoint> kpts_obj, kpts_scene;
	Detector->detect(img_gray_l, kpts_obj);  //keypoints 탐지
	Detector->detect(img_gray_r, kpts_scene);

	//특징점 시각화
	Mat img_kpts_l, img_kpts_r;
	drawKeypoints(img_gray_l, kpts_obj, img_kpts_l,
		Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_gray_r, kpts_scene, img_kpts_r,
		Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	imwrite("img_kpts_l.png", img_kpts_l);
	imwrite("img_kpts_r.png", img_kpts_r);

	//기술자(descriptor) 추출                             
	Ptr<SurfDescriptorExtractor> Extractor = SURF::create(100, 4, 3, false, true);
	Mat img_des_obj, img_des_scene;
	Extractor->compute(img_gray_l, kpts_obj, img_des_obj);
	Extractor->compute(img_gray_r, kpts_scene, img_des_scene);

	//추출한 descriptor를 이용하여 특징점 매칭
	BFMatcher matcher(NORM_L2); //NORM_L2로 거리 측정
	vector<DMatch> matches; 
	matcher.match(img_des_obj, img_des_scene, matches); //특징점 매칭 실행

	//매칭 결과 시각화
	Mat img_matches;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene, matches, img_matches, Scalar::all(-1),
		Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("img_matches.png", img_matches);

	//매칭 결과 정제
	//매칭 거리가 작은 우수한 매칭 결과를 정제하는 과정
	//최소 매칭 거리의 3배 또는 우수한 매칭 결과 60이상 까지 정제
	double dist_max = matches[0].distance;  //distance는 기술자 간의 거리
	double dist_min = matches[0].distance; 
	double dist;
	for (int i = 0; i < img_des_obj.rows; i++) {
		dist = matches[i].distance; //distance는 기술자 간의 거리
		if (dist < dist_min) dist_min = dist; // 매칭 거리가 작은 우수한 결과 찾기
		if (dist > dist_max) dist_max = dist;
	}
	printf("max_dist : %f\n", dist_max); //max는 사실상 불필요
	printf("min_dist : %f \n", dist_min); 

	//우수한 매칭결과 찾기
	vector<DMatch> matches_good;
	do {
		vector<DMatch> good_matches2;
		for (int i = 0; i < img_des_obj.rows; i++) {
			if (matches[i].distance < thresh_dist * dist_min) //최소매칭결과의 3배보다 작으면
				good_matches2.push_back(matches[i]); // 우수한 매칭결과
		}
		matches_good = good_matches2; 
		thresh_dist -= 1;
	} while (thresh_dist != 2 && matches_good.size() > min_matches); //우수한 매칭 결과 60이상까지

	//우수한 매칭결과 시각화
	Mat img_matches_good;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene, matches_good, img_matches_good,
		Scalar::all(-1),Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("img_matches_good.png", img_matches_good);

	//매칭 결과 좌표 추출
	vector<Point2f> obj, scene;
	for (int i = 0; i < matches_good.size(); i++) {
		obj.push_back(kpts_obj[matches_good[i].queryIdx].pt); //obj 기술자리스트에 저장된 인덱스의 좌표
		scene.push_back(kpts_scene[matches_good[i].trainIdx].pt); //scene 기술자리스트에 저장된 인덱스의 좌표
	}

	//매칭 결과로부터 homography 행렬 추출
	Mat mat_homo = findHomography(scene, obj, RANSAC);
	//이상치 제거를 위해 RANSAC추가
	//RANdom SAmple Consensus: 모델의 이상치들을 제거해 적절한 점들만 남기고 모델 추정

	//homograpy 행렬을 이용해 시점 역변환
	Mat img_result;
	warpPerspective(img_r, img_result, mat_homo, Size(img_l.cols * 2, img_l.rows * 1.2), INTER_CUBIC);
	//영상 잘림을 방지한 여유 size 부여

	//기존 영상과 역변환된 시점 영상 합체
	Mat img_pano;
	img_pano = img_result.clone();
	Mat roi(img_pano, Rect(0, 0, img_l.cols, img_l.rows)); //roi : 관심영역
	img_l.copyTo(roi); //합체

	Mat roi2(img_pano, Rect(img_l.cols - 10, 0, 20, img_l.rows));
	GaussianBlur(roi2, roi2, Size(5, 5), 0);

	//검은 여백 잘라내기
	int cut_x = 0, cut_y = 0;
	for (int y = 0; y < img_pano.rows; y++) {
		for (int x = 0; x < img_pano.cols; x++) {
			if (img_pano.at<Vec3b>(y, x)[0] == 0 &&
				img_pano.at<Vec3b>(y, x)[1] == 0 &&
				img_pano.at<Vec3b>(y, x)[2] == 0) {
				continue;
			}
			if (cut_x < x) cut_x = x;
			if (cut_y < y) cut_y = y;
		}
	}

	Mat img_pano_cut;
	img_pano_cut = img_pano(Range(0, cut_y), Range(0, cut_x));
	imwrite("img_pano_cut.png", img_pano_cut);

	return img_pano_cut; //최종 파노라마 이미지 return
}

void ex_panorama() {
	Mat matImg1 = imread("center.jpg", IMREAD_COLOR);
	Mat matImg2 = imread("left.jpg", IMREAD_COLOR);
	Mat matImg3 = imread("right.jpg", IMREAD_COLOR);


	if (matImg1.empty() || matImg2.empty() || matImg3.empty()) exit(-1);

	Mat result;
	flip(matImg1, matImg1, 1);
	flip(matImg2, matImg2, 1);
	result = makePanorama(matImg1, matImg2, 3, 60);
	flip(result, result, 1);
	result = makePanorama( result, matImg3, 3, 60);

	imshow("ex_panorama_result", result);
	imwrite("ex_panorama_result.png", result);
	waitKey();
}


void ex2(Mat book) { //scene에서 찾아낼 책을 인자로 받음
	Mat scene = imread("scene.jpg", 1);

	Mat sgray, bgray;
	//grayscale 변환
	cvtColor(scene, sgray, CV_BGR2GRAY);
	cvtColor(book, bgray, CV_BGR2GRAY);

	//SIFT를 이용하여 특징점 감지
	Ptr<SiftFeatureDetector>detector = SiftFeatureDetector::create(); 
	std::vector<KeyPoint> kpts_s,kpts_b;
	detector->detect(sgray, kpts_s); // scene에서 keypoints감지
	detector->detect(bgray, kpts_b); //book에서 keypoints감지


	//SIFT를 이용한 기술자 추출
	Ptr<SiftDescriptorExtractor> Extractor = SIFT::create(100, 4, 3, false, true);
	Mat img_des_book, img_des_scene; 
	Extractor->compute(bgray, kpts_b, img_des_book); 
	Extractor->compute(sgray, kpts_s, img_des_scene);
	
	//기술자를 이용한 특징점 매칭
	BFMatcher matcher(NORM_L2); //거리측정은 NORM_L2 방식 사용
	vector<DMatch> matches;
	matcher.match(img_des_book, img_des_scene, matches);

	//img_mathces에 매칭 결과 시각화
	Mat img_matches;
	drawMatches( bgray, kpts_b, sgray, kpts_s, matches, img_matches, Scalar::all(-1),
		Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	
//매칭 결과 정제
//매칭 거리가 작은 우수한 매칭 결과를 정제하는 과정
//최소 매칭 거리의 3배 또는 우수한 매칭 결과 60이상 까지 정제
	int thresh_dist = 3;
	int min_matches = 60;

	double dist_max = matches[0].distance; //distance는 기술자 간의 거리
	double dist_min = matches[0].distance;
	double dist;


	//가장 작은 매칭 거리 찾기
	for (int i = 0; i < img_des_book.rows; i++) { 
		dist = matches[i].distance;
		if (dist < dist_min) dist_min = dist;
		if (dist > dist_max) dist_max = dist;  //max는 사실상 불필요
	}
	printf("max_dist : %f\n", dist_max);
	printf("min_dist : %f \n", dist_min);

	//우수한 매칭 결과 찾기
	vector<DMatch> matches_good; 
	do {
		vector<DMatch> good_matches2;
		for (int i = 0; i < img_des_book.rows; i++) {
			if (matches[i].distance < thresh_dist * dist_min)  //최소매칭결과의 3배보다 작으면
				good_matches2.push_back(matches[i]); //우수 매칭 결과
		}
		matches_good = good_matches2;
		thresh_dist -= 1;
	} while (thresh_dist != 2 && matches_good.size() > min_matches);  //우수 매칭 결과 60이상이 될때까지

	//우수한 매칭결과 시각화
	Mat img_matches_good;
	drawMatches(bgray, kpts_b, sgray, kpts_s, matches_good, img_matches_good,
		Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


	//매칭 결과 좌표 추출
	vector<Point2f> b, s;
	for (int i = 0; i < matches_good.size(); i++) {
		b.push_back(kpts_b[matches_good[i].queryIdx].pt); //book 기술자리스트에 저장된 인덱스의 좌표
		s.push_back(kpts_s[matches_good[i].trainIdx].pt); //scene 기술자리스트에 저장된 인덱스 좌표
	}

	//매칭 결과로부터 homography 행렬 추출
	Mat mat_homo = findHomography(b, s, RANSAC);
	//이상치 제거를 위해 RANSAC추가

	//bgray 이미지 꼭짓점을 corner1에 넣고 호모그래피 행렬에 따라 투시변환
	vector<Point2f> corners1, corners2;
	corners1.push_back(Point2f(0, 0));
	corners1.push_back(Point2f(bgray.cols - 1, 0));
	corners1.push_back(Point2f(bgray.cols - 1, bgray.rows - 1));
	corners1.push_back(Point2f(0, bgray.rows - 1));
	perspectiveTransform(corners1, corners2, mat_homo);

	//호모그래피로 변환된 코너를 corners_dst에 넣기
	vector<Point> corners_dst;
	for (Point2f pt : corners2) 
		corners_dst.push_back(Point(cvRound(pt.x + bgray.cols), cvRound(pt.y)));

	//corners_dst로부터 사각형을 그림
	polylines(img_matches_good, corners_dst, true, Scalar(255,255,0), 2, LINE_AA);

	//결과 출력 및 저장
	imshow("img_matches_good.png", img_matches_good);
	imwrite("img_matches_good.png", img_matches_good);

	waitKey(0);
	destroyAllWindows();
}

int main() {
	/*ex1*/
	//ex_panorama_simple();
	ex_panorama();

	/*ex2*/
	//Mat book = imread("book1.jpg", 1);
	//Mat book = imread("book2.jpg", 1);
	//Mat book = imread("book3.jpg", 1);
	//ex2(book);

	return 0;
}