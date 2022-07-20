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
	//left,center,right�� �о�� imgs vector�� ����
	img = imread("left.jpg", IMREAD_COLOR);
	imgs.push_back(img);
	img = imread("center.jpg", IMREAD_COLOR);
	imgs.push_back(img);
	img = imread("right.jpg", IMREAD_COLOR);
	imgs.push_back(img);

	//stitcher class�� �̿��Ͽ� imgs stitch
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
	//grayscale�� ��ȯ
	Mat img_gray_l, img_gray_r;
	cvtColor(img_l, img_gray_l, CV_BGR2GRAY);
	cvtColor(img_r, img_gray_r, CV_BGR2GRAY);

	//SURF ������� Ư¡�� ����
	Ptr<SurfFeatureDetector> Detector = SURF::create(300);
	vector<KeyPoint> kpts_obj, kpts_scene;
	Detector->detect(img_gray_l, kpts_obj);  //keypoints Ž��
	Detector->detect(img_gray_r, kpts_scene);

	//Ư¡�� �ð�ȭ
	Mat img_kpts_l, img_kpts_r;
	drawKeypoints(img_gray_l, kpts_obj, img_kpts_l,
		Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_gray_r, kpts_scene, img_kpts_r,
		Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	imwrite("img_kpts_l.png", img_kpts_l);
	imwrite("img_kpts_r.png", img_kpts_r);

	//�����(descriptor) ����                             
	Ptr<SurfDescriptorExtractor> Extractor = SURF::create(100, 4, 3, false, true);
	Mat img_des_obj, img_des_scene;
	Extractor->compute(img_gray_l, kpts_obj, img_des_obj);
	Extractor->compute(img_gray_r, kpts_scene, img_des_scene);

	//������ descriptor�� �̿��Ͽ� Ư¡�� ��Ī
	BFMatcher matcher(NORM_L2); //NORM_L2�� �Ÿ� ����
	vector<DMatch> matches; 
	matcher.match(img_des_obj, img_des_scene, matches); //Ư¡�� ��Ī ����

	//��Ī ��� �ð�ȭ
	Mat img_matches;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene, matches, img_matches, Scalar::all(-1),
		Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("img_matches.png", img_matches);

	//��Ī ��� ����
	//��Ī �Ÿ��� ���� ����� ��Ī ����� �����ϴ� ����
	//�ּ� ��Ī �Ÿ��� 3�� �Ǵ� ����� ��Ī ��� 60�̻� ���� ����
	double dist_max = matches[0].distance;  //distance�� ����� ���� �Ÿ�
	double dist_min = matches[0].distance; 
	double dist;
	for (int i = 0; i < img_des_obj.rows; i++) {
		dist = matches[i].distance; //distance�� ����� ���� �Ÿ�
		if (dist < dist_min) dist_min = dist; // ��Ī �Ÿ��� ���� ����� ��� ã��
		if (dist > dist_max) dist_max = dist;
	}
	printf("max_dist : %f\n", dist_max); //max�� ��ǻ� ���ʿ�
	printf("min_dist : %f \n", dist_min); 

	//����� ��Ī��� ã��
	vector<DMatch> matches_good;
	do {
		vector<DMatch> good_matches2;
		for (int i = 0; i < img_des_obj.rows; i++) {
			if (matches[i].distance < thresh_dist * dist_min) //�ּҸ�Ī����� 3�躸�� ������
				good_matches2.push_back(matches[i]); // ����� ��Ī���
		}
		matches_good = good_matches2; 
		thresh_dist -= 1;
	} while (thresh_dist != 2 && matches_good.size() > min_matches); //����� ��Ī ��� 60�̻����

	//����� ��Ī��� �ð�ȭ
	Mat img_matches_good;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene, matches_good, img_matches_good,
		Scalar::all(-1),Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("img_matches_good.png", img_matches_good);

	//��Ī ��� ��ǥ ����
	vector<Point2f> obj, scene;
	for (int i = 0; i < matches_good.size(); i++) {
		obj.push_back(kpts_obj[matches_good[i].queryIdx].pt); //obj ����ڸ���Ʈ�� ����� �ε����� ��ǥ
		scene.push_back(kpts_scene[matches_good[i].trainIdx].pt); //scene ����ڸ���Ʈ�� ����� �ε����� ��ǥ
	}

	//��Ī ����κ��� homography ��� ����
	Mat mat_homo = findHomography(scene, obj, RANSAC);
	//�̻�ġ ���Ÿ� ���� RANSAC�߰�
	//RANdom SAmple Consensus: ���� �̻�ġ���� ������ ������ ���鸸 ����� �� ����

	//homograpy ����� �̿��� ���� ����ȯ
	Mat img_result;
	warpPerspective(img_r, img_result, mat_homo, Size(img_l.cols * 2, img_l.rows * 1.2), INTER_CUBIC);
	//���� �߸��� ������ ���� size �ο�

	//���� ����� ����ȯ�� ���� ���� ��ü
	Mat img_pano;
	img_pano = img_result.clone();
	Mat roi(img_pano, Rect(0, 0, img_l.cols, img_l.rows)); //roi : ���ɿ���
	img_l.copyTo(roi); //��ü

	Mat roi2(img_pano, Rect(img_l.cols - 10, 0, 20, img_l.rows));
	GaussianBlur(roi2, roi2, Size(5, 5), 0);

	//���� ���� �߶󳻱�
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

	return img_pano_cut; //���� �ĳ�� �̹��� return
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


void ex2(Mat book) { //scene���� ã�Ƴ� å�� ���ڷ� ����
	Mat scene = imread("scene.jpg", 1);

	Mat sgray, bgray;
	//grayscale ��ȯ
	cvtColor(scene, sgray, CV_BGR2GRAY);
	cvtColor(book, bgray, CV_BGR2GRAY);

	//SIFT�� �̿��Ͽ� Ư¡�� ����
	Ptr<SiftFeatureDetector>detector = SiftFeatureDetector::create(); 
	std::vector<KeyPoint> kpts_s,kpts_b;
	detector->detect(sgray, kpts_s); // scene���� keypoints����
	detector->detect(bgray, kpts_b); //book���� keypoints����


	//SIFT�� �̿��� ����� ����
	Ptr<SiftDescriptorExtractor> Extractor = SIFT::create(100, 4, 3, false, true);
	Mat img_des_book, img_des_scene; 
	Extractor->compute(bgray, kpts_b, img_des_book); 
	Extractor->compute(sgray, kpts_s, img_des_scene);
	
	//����ڸ� �̿��� Ư¡�� ��Ī
	BFMatcher matcher(NORM_L2); //�Ÿ������� NORM_L2 ��� ���
	vector<DMatch> matches;
	matcher.match(img_des_book, img_des_scene, matches);

	//img_mathces�� ��Ī ��� �ð�ȭ
	Mat img_matches;
	drawMatches( bgray, kpts_b, sgray, kpts_s, matches, img_matches, Scalar::all(-1),
		Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	
//��Ī ��� ����
//��Ī �Ÿ��� ���� ����� ��Ī ����� �����ϴ� ����
//�ּ� ��Ī �Ÿ��� 3�� �Ǵ� ����� ��Ī ��� 60�̻� ���� ����
	int thresh_dist = 3;
	int min_matches = 60;

	double dist_max = matches[0].distance; //distance�� ����� ���� �Ÿ�
	double dist_min = matches[0].distance;
	double dist;


	//���� ���� ��Ī �Ÿ� ã��
	for (int i = 0; i < img_des_book.rows; i++) { 
		dist = matches[i].distance;
		if (dist < dist_min) dist_min = dist;
		if (dist > dist_max) dist_max = dist;  //max�� ��ǻ� ���ʿ�
	}
	printf("max_dist : %f\n", dist_max);
	printf("min_dist : %f \n", dist_min);

	//����� ��Ī ��� ã��
	vector<DMatch> matches_good; 
	do {
		vector<DMatch> good_matches2;
		for (int i = 0; i < img_des_book.rows; i++) {
			if (matches[i].distance < thresh_dist * dist_min)  //�ּҸ�Ī����� 3�躸�� ������
				good_matches2.push_back(matches[i]); //��� ��Ī ���
		}
		matches_good = good_matches2;
		thresh_dist -= 1;
	} while (thresh_dist != 2 && matches_good.size() > min_matches);  //��� ��Ī ��� 60�̻��� �ɶ�����

	//����� ��Ī��� �ð�ȭ
	Mat img_matches_good;
	drawMatches(bgray, kpts_b, sgray, kpts_s, matches_good, img_matches_good,
		Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


	//��Ī ��� ��ǥ ����
	vector<Point2f> b, s;
	for (int i = 0; i < matches_good.size(); i++) {
		b.push_back(kpts_b[matches_good[i].queryIdx].pt); //book ����ڸ���Ʈ�� ����� �ε����� ��ǥ
		s.push_back(kpts_s[matches_good[i].trainIdx].pt); //scene ����ڸ���Ʈ�� ����� �ε��� ��ǥ
	}

	//��Ī ����κ��� homography ��� ����
	Mat mat_homo = findHomography(b, s, RANSAC);
	//�̻�ġ ���Ÿ� ���� RANSAC�߰�

	//bgray �̹��� �������� corner1�� �ְ� ȣ��׷��� ��Ŀ� ���� ���ú�ȯ
	vector<Point2f> corners1, corners2;
	corners1.push_back(Point2f(0, 0));
	corners1.push_back(Point2f(bgray.cols - 1, 0));
	corners1.push_back(Point2f(bgray.cols - 1, bgray.rows - 1));
	corners1.push_back(Point2f(0, bgray.rows - 1));
	perspectiveTransform(corners1, corners2, mat_homo);

	//ȣ��׷��Ƿ� ��ȯ�� �ڳʸ� corners_dst�� �ֱ�
	vector<Point> corners_dst;
	for (Point2f pt : corners2) 
		corners_dst.push_back(Point(cvRound(pt.x + bgray.cols), cvRound(pt.y)));

	//corners_dst�κ��� �簢���� �׸�
	polylines(img_matches_good, corners_dst, true, Scalar(255,255,0), 2, LINE_AA);

	//��� ��� �� ����
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