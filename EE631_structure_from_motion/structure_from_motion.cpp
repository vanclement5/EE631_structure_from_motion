#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <fstream>

using namespace cv;
using namespace std;

void tasks(char *path_in, char *path_out);

int main(int argc, char **argv) {
	char path[50];
	Mat optical_flow[17];

	for (int i = 1; i < 17; i++) {
		sprintf(path, "Optical Flow\\O%d.jpg", i);
		optical_flow[i - 1] = imread(path, IMREAD_GRAYSCALE);
	}

	char path_in[50];
	char path_out[50];
	sprintf(path_in, "Parallel Cube\\ParallelCube");
	sprintf(path_out, "parallel_cube");
	tasks(path_in, path_out);

	sprintf(path_in, "Parallel Real\\ParallelReal");
	sprintf(path_out, "parallel_real");
	tasks(path_in, path_out);

	sprintf(path_in, "Turned Cube\\TurnCube");
	sprintf(path_out, "turned_cube");
	tasks(path_in, path_out);

	sprintf(path_in, "Turned Real\\TurnReal");
	sprintf(path_out, "turned_real");
	tasks(path_in, path_out);



}


void tasks(char *path_in, char *path_out) {
	char path[50];
	Mat parallel_cube[17];

	for (int i = 10; i < 16; i++) {
		sprintf(path, "%s%d.jpg", path_in, i);
		parallel_cube[i - 10] = imread(path, IMREAD_GRAYSCALE);
	}


	int max_corners = 300;
	double quality_level = 0.01;
	double min_distance = 4;

	Mat prev_pts;
	Mat err;
	Mat out[2];
	int skip = 1;
	int width = 40;
	int height = 40;
	int swidth = 80;
	int sheight = 80;
	Mat pt_status[5];

	vector<vector<Point2f>> corners_next(17, vector<Point2f>(300));
	goodFeaturesToTrack(parallel_cube[0], corners_next[0], max_corners, quality_level, min_distance);
	for (int i = 0; i < 5; i++){
		printf("Current Index: %d\n", i);
		for (int j = 0; j < corners_next[i].size(); j++) {
			int x = corners_next[i][j].x - width / 2;
			int y = corners_next[i][j].y - height / 2;

			int sx = corners_next[i][j].x - swidth / 2;
			int sy = corners_next[i][j].y - sheight / 2;

			Mat match_output;
			double min_val, max_val;
			Point min_idx, max_idx;

			if (x + width - 1 > 639 || sx + swidth - 1 > 639){
				corners_next[i + 1][j] = Point(-1, -1);
				continue;
			}
			else if (x < 0 || sx < 0) {
				corners_next[i + 1][j] = Point(-1, -1);
				continue;
			}
			if (y + height - 1 > 479 || sy + sheight - 1 > 479) {
				corners_next[i + 1][j] = Point(-1, -1);
				continue;
			}
			else if (y < 0 || sy < 0) {
				corners_next[i + 1][j] = Point(-1, -1);
				continue;
			}

			matchTemplate(parallel_cube[i + 1](Rect(sx, sy, swidth, sheight)), parallel_cube[i](Rect(x, y, width, height)), match_output, CV_TM_SQDIFF);
			minMaxLoc(match_output, &min_val, &max_val, &min_idx, &max_idx);

			Point match_idx;
			match_idx.x = min_idx.x + sx + width / 2;
			match_idx.y = min_idx.y + sy + height / 2;
			corners_next[i + 1][j] = match_idx;
		}
		findFundamentalMat(corners_next[i], corners_next[i + 1], CV_FM_RANSAC, 3., 0.99, pt_status[i]);
	}

	Mat keepers;
	bitwise_and(pt_status[0], pt_status[1], keepers);
	for (int i = 2; i < 5; i++) {
		bitwise_and(pt_status[i], keepers, keepers);
	}



	Mat status, F, H1, H2, M, Minv, R1, R2, distCoeff1, distCoeff2, map11,map12,map21,map22;
	vector<vector<Point2f>> points(2, vector<Point2f>(0));

	F = findFundamentalMat(corners_next[0], corners_next[5], CV_FM_RANSAC, 3., 0.999, status);
	uchar *st = status.ptr<uchar>(0);
	float diff;

	for (int i = 0; i < corners_next[0].size(); i++) {
		if (st[i] == 0){
			continue;
		}
		diff = (sqrt((corners_next[0][i].x - corners_next[5][i].x)*(corners_next[0][i].x - corners_next[5][i].x) +
			(corners_next[0][i].y - corners_next[5][i].y)*(corners_next[5][i].y - corners_next[5][i].y)));
		if (diff > 100)
			continue;

		points[0].push_back(corners_next[0][i]);
		points[1].push_back(corners_next[5][i]);
	}



	cvtColor(parallel_cube[0], out[0], CV_GRAY2BGR);
	cvtColor(parallel_cube[5], out[1], CV_GRAY2BGR);

	for (int j = 0; j < points[0].size(); j++) {
		line(out[0], points[0][j], points[1][j], Scalar(0, 0, 255), 2);
	}
	for (int j = 0; j < points[0].size(); j++){
		circle(out[0], points[0][j], 1, Scalar(0, 255, 0), -1);
	}


	for (int j = 0; j < points[1].size(); j++){
		circle(out[1], points[1][j], 1, Scalar(0, 255, 0), -1);
	}

	// do task 1 output
	F = findFundamentalMat(points[0], points[1], CV_FM_RANSAC, 3., 0.9999, status);
	stereoRectifyUncalibrated(points[0], points[1], F, Size(640, 480), H1, H2,10);

	M = (Mat_<double>(3, 3) << 2000, 0, 320, 0, 2000, 240, 0, 0, 1);
	Minv = M.inv();
	R1 = Minv*H1*M;
	R2 = Minv*H2*M;

	distCoeff1 = (Mat_<double>(1, 5) << 0,0,0,0,0);

	initUndistortRectifyMap(M, distCoeff1, R1, M, Size(640, 480), CV_32FC1, map11, map12);
	initUndistortRectifyMap(M, distCoeff1, R2, M, Size(640, 480), CV_32FC1, map21, map22);

	Mat ctd1, ctd2;
	remap(parallel_cube[0], ctd1, map11, map12, INTER_LINEAR);
	remap(parallel_cube[5], ctd2, map21, map22, INTER_LINEAR);

	cvtColor(ctd1, ctd1, CV_GRAY2BGR);
	cvtColor(ctd2, ctd2, CV_GRAY2BGR);

	for (int i = 0; i < 480; i += 20) {
		line(ctd1, Point2f(0, i), Point2f(639, i), Scalar(0, 255, 0));
		line(ctd2, Point2f(0, i), Point2f(639, i), Scalar(0, 255, 0));
	}

	char full_path_out[50];
	sprintf(full_path_out, "%s1.png", path_out);
	imwrite(full_path_out, ctd1);
	sprintf(full_path_out, "%s2.png", path_out);
	imwrite(full_path_out, ctd2);

	// do task2 output
	M = (Mat_<double>(3, 3) << 825, 0, 331.653, 0, 824.267, 252.928, 0, 0, 1);
	distCoeff1 = (Mat_<double>(1, 5) << -0.23807, 0.093132, 0.000324, -0.002190, 0.46417);

	vector<vector<Point2f>> pointsctd(2, vector<Point2f>(0));
	undistortPoints(points[0],pointsctd[0],M,distCoeff1);
	undistortPoints(points[1], pointsctd[1], M, distCoeff1);

	// convert back to image coordinates
	for (int i = 0; i < pointsctd[0].size(); i++) {
		pointsctd[0][i].x = pointsctd[0][i].x * 825 + 331.653;
		pointsctd[0][i].y = pointsctd[0][i].y * 824.267 + 252.928;
		pointsctd[1][i].x = pointsctd[1][i].x * 825 + 331.653;
		pointsctd[1][i].y = pointsctd[1][i].y * 824.267 + 252.928;
	}

	F = findFundamentalMat(pointsctd[0], pointsctd[1], CV_FM_RANSAC, 3., 0.9999, status);
	Mat E = M.t() * F * M;

	Mat t, RR, TR, RL, TL, R, T;
	Mat w, u, vt, R0, R3, T0, T1, T2, T3;
	SVD::compute(E, w, u, vt);
	w = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 0);
	E = u*w*vt;
	recoverPose(E, pointsctd[0], pointsctd[1], R, T, 825, Point2d(332, 253));

	sprintf(full_path_out, "%s_params.txt", path_out);
	ofstream ofs;

	ofs.open(full_path_out, ofstream::out);
	ofs << "R = " << endl << " " << R << endl << endl;
	ofs << "T = " << endl << " " << T << endl << endl;
	ofs << "E = " << endl << " " << E << endl << endl;
	ofs << "F = " << endl << " " << F << endl << endl;
	ofs.close();

	// task 3 code
	Mat P1, P2, Q, im1, im2, map1, map2, pts4d;

	vector<Point3f> p3d;
	for (int i = 0; i < 160; i++) {
		Mat pt0 = (Mat_<double>(3, 1) << pointsctd[0][i].x, pointsctd[0][i].y, 1);
		Mat pt1 = (Mat_<double>(3, 1) << pointsctd[1][i].x, pointsctd[1][i].y, 1);

		Mat x = M.inv()*pt0;
		Mat x1 = -R.t() * M.inv()*pt1;
		Mat x2 = (M.inv()*pt0).cross(R.t()*M.inv()*pt1);
		Mat B = -R.t()*T;
		Mat tmp, A, X;
		hconcat(x, x1, tmp);
		hconcat(tmp, x2, A);
		solve(A, B, X);
		double a = X.at<double>(0, 0);
		double b = X.at<double>(1, 0);
		double x_0 = A.at<double>(0, 0);
		double x_1 = -A.at<double>(0,1);
		double y_0 = A.at<double>(1,0);
		double y_1 = -A.at<double>(1, 1);
		double z_0 = A.at<double>(2,0);
		double z_1 = -A.at<double>(2,1);

		Point3f pt;
		pt.x = (a*x_0 + b*x_1)/2;
		pt.y = (a*y_0 + b*y_1) / 2;
		pt.z = (a*z_0 + b*z_1) / 2;
		p3d.push_back(pt);
	}

	double min = 10e6;
	for (int i = 0; i < p3d.size(); i++) {
		double d = p3d[i].z;
		if (d > 0 && d < min){
			min = d;
		}
	}
	double scale = 20 / min;

	int start, stop;
	if (!strcmp(path_out, "parallel_cube")) {
		scale = 1.69;
		start = 0;
		stop = 4;
	}
	else if (!strcmp(path_out, "parallel_real")) {
		scale = 1.69;
		start = 10;
		stop = 14;
	}
	else if (!strcmp(path_out, "turned_cube")) {
		scale = 2.09;
		start = 10;
		stop = 14;
	}
	else if (!strcmp(path_out, "turned_real")) {
		scale = 2.09;
		start = 50;
		stop = 54;
	}
	parallel_cube[0].copyTo(out[0]);

	cvtColor(out[0], out[0], CV_GRAY2BGR);
	for (int i = start; i < stop; i++) {
		circle(out[0], pointsctd[0][i], 10, Scalar(0, 255, 0));
		char text[50];
		sprintf(text, "%.02f, %.02f, %.02f", scale*p3d[i].x, scale*p3d[i].y, scale*p3d[i].z);
		putText(out[0], text, Point(pointsctd[0][i].x + 5, pointsctd[0][i].y + 5), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255), 2);
	}
	sprintf(full_path_out, "%s3.png", path_out);
	imwrite(full_path_out, out[0]);
}
