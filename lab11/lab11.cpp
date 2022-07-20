#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

#define _USE_MATH_DEFINES

using namespace std;
using namespace cv;

int ex1() {
	//random colors 생성
	vector<Scalar> colors;
	RNG rng;
	for (int i = 0; i < 100; i++) {
		int r = rng.uniform(0, 256);
		int g = rng.uniform(0, 256);
		int b = rng.uniform(0, 256);
		colors.push_back(Scalar(r, g, b));
	}

	Mat old_frame, old_gray;
	vector<Point2f> p0, p1;

	//Take first frame and find corners in it
	old_frame = imread("1.jpg", 1);
	//old_frame = imread("111.jpg",1); //첫번째 영상 입력
	cvtColor(old_frame, old_gray, COLOR_BGR2GRAY); //grayscale 변환
	goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04); //코너점을 찾는 함수

	//Create a mask image for drawing purposes
	Mat mask = Mat::zeros(old_frame.size(), old_frame.type());

	while (true) {
		Mat frame, frame_gray;
		frame = imread("2.jpg", 1); //두번 영상입력
		//frame = imread("222.jpg", 1); //두번 영상입력
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY); //grayscale 변환

		// optical flow 계산
		vector<uchar> status;
		vector<float> err;
		TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
		calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15, 15), 2, criteria);

		Mat dst;
		addWeighted(frame, 0.5, old_frame, 0.5, 0,dst); //두영상을 0,5 ,0,5로 blending

		vector<Point2f> good_new;
		for (uint i = 0; i < p0.size(); i++) {
			//good points 선택
			if (status[i] == 1) {
				good_new.push_back(p1[i]);
				//어느정도 이동했는지 표시
				line(mask, p1[i], p0[i], colors[i], 2);
				circle(frame, p1[i], 5, colors[i], -1);
			}
		}
		Mat img;
		add(dst, mask, img); //dst과 mask합치기

		imshow("Frame", img);

		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27)
			break;

		//Now update the previous frame and previous points
		old_gray = frame_gray.clone();
		p0 = good_new;

	}
	return 0;
}



		/*파네벡 알고리즘*/
int ex2() {
	VideoCapture capture(samples::findFile("test.mp4"));
	if (!capture.isOpened()) {
		//error in opening the video input
		cerr << "Unable to open file!" << endl;
		return 0;
	}
	
	Mat frame1, prvs;
	capture >> frame1; //frame1 캡처
	cvtColor(frame1, prvs, COLOR_BGR2GRAY); //grayscale로 변환

	while (true) {
		Mat frame2, next;
		capture >> frame2; //frame2 캡쳐
		if (frame2.empty()) break;
		cvtColor(frame2, next, COLOR_BGR2GRAY); //grayscale 변환

		Mat flow(prvs.size(), CV_32FC2);
		calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0); //파네백 알고리즘 수행
	
		/*격자안에서 모션 벡터 표시하기*/
	
		for (int y = 0; y < frame2.rows; y += 15) {
			for (int x = 0; x < frame2.cols; x += 15) {
				const Point2f flowatxy = flow.at<Point2f>(y, x); 
				/*변화량만큼 모션벡터 표시하기*/
				line(frame2, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar( 0,255, 0),2,LINE_AA);
				circle(frame2, Point(x, y), 1, Scalar(0, 255, 0), -1);
			}
		}
		imshow("frame2", frame2);

		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27) break;

		prvs = next;
	}
	return 0;
}

int main() {
	ex2();
}
