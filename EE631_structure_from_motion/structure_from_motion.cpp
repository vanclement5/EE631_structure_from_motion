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
	int width = 15;
	int height = 15;
	int swidth = 50;
	int sheight = 50;
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
	sprintf(full_path_out, "%s1.bmp", path_out);
	imwrite(full_path_out, ctd1);
	sprintf(full_path_out, "%s2.bmp", path_out);
	imwrite(full_path_out, ctd2);


	// do task2 output
	M = (Mat_<double>(3, 3) << 825, 0, 331.653, 0, 824.267, 252.928, 0, 0, 1);
	distCoeff1 = (Mat_<double>(1, 5) <<-0.23807, 0.093132, 0.000324, -0.002190, 0.46417);
	Mat E = M.t() * F * M;

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

	Mat w, u, vt, R0, R3, T0, T1, T2, T3;
	SVD::compute(E, w, u, vt);
	w = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 0);
	
	Mat Rz0 = (Mat_<double>(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
	Mat Rz1 = (Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);

	R0 = u*Rz0.t() * vt;
	R1 = u*Rz1.t() * vt;

	T0 = u*Rz0*w*u.t();
	T1 = u*Rz1*w*u.t();

	Mat t0 = (Mat_<double>(3, 1) << T0.at<double>(2, 1), T0.at<double>(2, 0), T0.at<double>(1, 0));
	Mat t1 = (Mat_<double>(3, 1) << T1.at<double>(2, 1), T1.at<double>(2, 0), T1.at<double>(1, 0));

	Mat p3d0[300];
	Mat p3d1[300];
	Mat p3d2[300];
	Mat p3d3[300];

	for (int i = 0; i < 50; i++) {
		Mat pt0 = (Mat_<double>(3, 1) << pointsctd[0][i].x, pointsctd[0][i].y, 1);
		Mat pt1 = (Mat_<double>(3, 1) << pointsctd[1][i].x, pointsctd[1][i].y, 1);

		Mat x = M.inv()*pt0;
		Mat x1 = -R0.t() * M.inv()*pt1;
		Mat x2 = M.inv()*pt0.cross(R0.t()*M.inv()*pt1);
		Mat B = -R0.t()*t0;
		Mat tmp, A, X;
		hconcat(x, x1, tmp);
		hconcat(tmp, x2, A);
		solve(A, B, X);
		tmp = (Mat_<double>(3, 1) << (X.at<double>(0, 0)*A.at<double>(0, 0) - X.at<double>(1, 0)*A.at<double>(0, 1)) / 2,
			(X.at<double>(0, 0)*A.at<double>(1, 0) - X.at<double>(1, 0)*A.at<double>(1, 1)) / 2,
			(X.at<double>(0, 0)*A.at<double>(2, 0) - X.at<double>(1, 0)*A.at<double>(2, 1)) / 2);
		p3d0[i] = tmp;

		x = M.inv()*pt0;
		x1 = -R0.t() * M.inv()*pt1;
		x2 = M.inv()*pt0.cross(R0.t()*M.inv()*pt1);
		B = -R0.t()*t1;
		tmp, A, X;
		hconcat(x, x1, tmp);
		hconcat(tmp, x2, A);
		solve(A, B, X);
		tmp = (Mat_<double>(3, 1) << (X.at<double>(0, 0)*A.at<double>(0, 0) - X.at<double>(1, 0)*A.at<double>(0, 1)) / 2,
			(X.at<double>(0, 0)*A.at<double>(1, 0) - X.at<double>(1, 0)*A.at<double>(1, 1)) / 2,
			(X.at<double>(0, 0)*A.at<double>(2, 0) - X.at<double>(1, 0)*A.at<double>(2, 1)) / 2);
		p3d1[i] = tmp;

		x = M.inv()*pt0;
		x1 = -R1.t() * M.inv()*pt1;
		x2 = M.inv()*pt0.cross(R1.t()*M.inv()*pt1);
		B = -R1.t()*t0;
		tmp, A, X;
		hconcat(x, x1, tmp);
		hconcat(tmp, x2, A);
		solve(A, B, X);
		tmp = (Mat_<double>(3, 1) << (X.at<double>(0, 0)*A.at<double>(0, 0) - X.at<double>(1, 0)*A.at<double>(0, 1)) / 2,
			(X.at<double>(0, 0)*A.at<double>(1, 0) - X.at<double>(1, 0)*A.at<double>(1, 1)) / 2,
			(X.at<double>(0, 0)*A.at<double>(2, 0) - X.at<double>(1, 0)*A.at<double>(2, 1)) / 2);
		p3d2[i] = tmp;

		x = M.inv()*pt0;
		x1 = -R1.t() * M.inv()*pt1;
		x2 = M.inv()*pt0.cross(R1.t()*M.inv()*pt1);
		B = -R1.t()*t1;
		tmp, A, X;
		hconcat(x, x1, tmp);
		hconcat(tmp, x2, A);
		solve(A, B, X);
		tmp = (Mat_<double>(3, 1) << (X.at<double>(0, 0)*A.at<double>(0, 0) - X.at<double>(1, 0)*A.at<double>(0, 1)) / 2,
			(X.at<double>(0, 0)*A.at<double>(1, 0) - X.at<double>(1, 0)*A.at<double>(1, 1)) / 2,
			(X.at<double>(0, 0)*A.at<double>(2, 0) - X.at<double>(1, 0)*A.at<double>(2, 1)) / 2);
		p3d3[i] = tmp;
	}

	int use0 = 1;
	int use1 = 1;
	int use2 = 1;
	int use3 = 1;

	for (int i = 0; i < 50; i++) {
		if (p3d0[i].at<double>(2, 0) < 0)
			use0--;
		if (p3d1[i].at<double>(2, 0) < 0)
			use1--;
		if (p3d2[i].at<double>(2, 0) < 0)
			use2--;
		if (p3d3[i].at<double>(2, 0) < 0)
			use3--;
	}

	Mat R_final, T_final;
	Mat *p3d;
	if (use0 + use1 + use2 + use3 > 1) {
		printf("Multiple positives...\n");
	}
	else if (use0 <=0 &&  use1 <= 0 &&  use2 <=0 && use3 <= 0) {
		printf("No positives...\n");
		if (use0 > use1 && use0 > use2 && use0 > use3) {
			use0 = 1;
		}
		else if (use1 > use0 && use1 > use2 && use1 > use3) {
			use1 = 1;
		}
		else if (use2 > use1 && use2 > use0 && use2 > use3) {
			use2 = 1;
		}
		else if (use3 > use1 && use3 > use2 && use3 > use0) {
			use3 = 1;
		}
	}

	if (use0==1) {
		R_final = R0;
		T_final = t0;
		p3d = p3d0;
	}
	else if (use1 == 1) {
		R_final = R0;
		T_final = t1;
		p3d = p3d1;
	}
	else if (use2 == 1) {
		R_final = R1;
		T_final = t0;
		p3d = p3d2;
	}
	else if (use3 == 1) {
		R_final = R1;
		T_final = t1;
		p3d = p3d3;
	}
	sprintf(full_path_out, "%s_params.txt", path_out);

	ofstream ofs;

	ofs.open(full_path_out, ofstream::out);
	ofs << "R = " << endl << " " << R_final << endl << endl;
	ofs << "T = " << endl << " " << T_final << endl << endl;
	ofs << "E = " << endl << " " << E << endl << endl;
	ofs << "F = " << endl << " " << F << endl << endl;
	ofs.close();

	// task 3 code
	double min = 100;
	for (int i = 0; i < 50; i++) {
		double d = p3d[i].at<double>(2, 0);
		if (d > 0 && d < min){
			min = d;
		}
	}
	double scale = 20 / min;

	for (int i = 0; i < 4; i++) {
		circle(out[0], pointsctd[0][i * 6], 10, Scalar(0, 255, 0));
		char text[50];
		sprintf(text, "%.02f, %.02f, %.02f", scale*p3d[i].at<double>(0, 0), scale*p3d[i].at<double>(1, 0), scale*p3d[i].at<double>(2, 0));
		putText(out[0], text, Point(pointsctd[0][i * 6].x + 5, pointsctd[0][i * 6].y + 5), FONT_HERSHEY_PLAIN,1,Scalar(0,255,0),2);
	}


}
