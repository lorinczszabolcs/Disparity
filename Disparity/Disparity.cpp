// Disparity.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>
#include <fstream>
#include <cmath>
#include <opencv\cv.hpp>

typedef cv::Point_<float_t> Pixel;

int getMatchingPoint(
	const cv::Point& p_,		// the point whose neighborhood we test for matching points
	const cv::Mat& left_,		// left image
	const cv::Mat& right_,		// right image
	const int& window_size_,	// window size
	const int& max_disp_);		// maximum disparity

void naiveDisparityMap(
	const cv::Mat& left_,		// left image
	const cv::Mat& right_,		// right image
	const int& window_size_,	// window size
	const int& max_disp_,		// maximum disparity
	cv::Mat& disp_);			// disparity

void dynamicDisparityMap(
	const cv::Mat& left_,		// left image
	const cv::Mat& right_,		// right image
	const int& window_size_,	// window size
	const float& lambda_,		// maximum disparity
	cv::Mat& disp_);			// disparity

void cameraToWorld(
	const cv::Mat& disp_,		// disparity map
	const float& dmin_,			// additional parameter for shifting
	const float& f_,			// focal length
	const float& b_,			// baseline
	cv::Mat& world_);			// world coordinates

int main(int argc, char* argv[])
{
	// calibration parameters
	std::ifstream infile("D:\\Learning\\University\\MSc\\1styear\\1stSemester\\3D Sensing and Sensor Fusion\\Reindeer-2views\\Reindeer\\dmin.txt");
	std::ofstream naive_outfile("D:\\Learning\\University\\MSc\\1styear\\1stSemester\\3D Sensing and Sensor Fusion\\naive.xyz");
	std::ofstream dyn_outfile("D:\\Learning\\University\\MSc\\1styear\\1stSemester\\3D Sensing and Sensor Fusion\\dyn.xyz");

	float dmin;
	infile >> dmin;
	double f = 3740.;
	double b = 160.;

	// Reading images
	cv::Mat left = cv::imread("D:\\Learning\\University\\MSc\\1styear\\1stSemester\\3D Sensing and Sensor Fusion\\Reindeer-2views\\Reindeer\\view1.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat right = cv::imread("D:\\Learning\\University\\MSc\\1styear\\1stSemester\\3D Sensing and Sensor Fusion\\Reindeer-2views\\Reindeer\\view5.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat gt_disp = cv::imread("D:\\Learning\\University\\MSc\\1styear\\1stSemester\\3D Sensing and Sensor Fusion\\Reindeer-2views\\Reindeer\\disp1.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat naive_disp, dyn_disp;
	cv::Mat naive_normalized_disp, dyn_normalized_disp;


	// Rescaling for faster prototyping
	float scale = .25;
	cv::resize(left, left, cv::Size(), scale, scale);
	cv::resize(right, right, cv::Size(), scale, scale);
	
	std::cout << "Ready loading! \n";

	// Initializing windows and showing original images
	cv::namedWindow("Left image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Left image", left);

	cv::namedWindow("Right image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Right image", right);

	// reading arguments
	int naive_window_size = atoi(argv[1]);
	int dyn_window_size = atoi(argv[2]);
	int max_disp = atoi(argv[3]);
	float lambda = atoi(argv[4]);
	
	// Calculating and showing naive disparity image
	naiveDisparityMap(left, right, naive_window_size, max_disp, naive_disp);
	cv::namedWindow("Naive Disparity", cv::WINDOW_AUTOSIZE);
	cv::normalize(naive_disp, naive_normalized_disp, 255, 0, cv::NORM_MINMAX);
	cv::imshow("Naive Disparity", naive_normalized_disp);
	
	// Calculating and showing dynamic disparity image
	dynamicDisparityMap(left, right, dyn_window_size, lambda, dyn_disp);
	cv::namedWindow("Dyn Disparity", cv::WINDOW_AUTOSIZE);
	cv::normalize(dyn_disp, dyn_normalized_disp, 255, 0, cv::NORM_MINMAX);
	cv::imshow("Dyn Disparity", dyn_normalized_disp);

	// Calculating naive world coordinates
	cv::Mat naive_world(naive_disp.size(), CV_32FC3);
	cameraToWorld(naive_disp, dmin, f, b, naive_world);

	std::cout << "Naive world calc ready!" << "\n";

	// Calculating dynamic world coordinates
	cv::Mat dyn_world(naive_disp.size(), CV_32FC3);
	cameraToWorld(naive_disp, dmin, f, b, dyn_world);

	std::cout << "Dyn world calc ready!" << "\n";
	
	// writing naive xyz
	for (size_t i = naive_window_size; i < naive_world.rows - naive_window_size; i++)
	{
		for (size_t j = naive_window_size; j < naive_world.cols - naive_window_size; j++)
		{
			if (naive_disp.at<int>(i, j) != 0)
			{
				naive_outfile << naive_world.at<cv::Vec3f>(i, j)[0] << " " << naive_world.at<cv::Vec3f>(i, j)[1] << " " << naive_world.at<cv::Vec3f>(i, j)[2] << "\n";
			}
		}
	}
	naive_outfile.close();

	std::cout << "Naive world out ready!" << "\n";

	// writing dynamic xyz
	for (size_t i = naive_window_size; i < dyn_world.rows - naive_window_size; i++)
	{
		for (size_t j = naive_window_size; j < dyn_world.cols - naive_window_size; j++)
		{
			if (dyn_disp.at<int>(i, j) != 0)
			{
				dyn_outfile << dyn_world.at<cv::Vec3f>(i, j)[0] << " " << dyn_world.at<cv::Vec3f>(i, j)[1] << " " << dyn_world.at<cv::Vec3f>(i, j)[2] << "\n";
			}
		}
	}
	naive_outfile.close();
	
	std::cout << "Dyn world out ready!" << "\n";
	
	// Waiting
	cv::waitKey();
}

// Get matching point for p from left_ image in right_image using window of size window_size_
int getMatchingPoint(
	const cv::Point& p_,		// the point whose neighborhood we test for matching points
	const cv::Mat& left_,		// left image
	const cv::Mat& right_,		// right image
	const int& window_size_,	// window size
	const int& max_disp_)		// maximum disparity
{	
	// initialization
	long min_sad = std::numeric_limits<long>::max();
	int min_idx = 0;
	cv::Mat abs_diff;

	// left block
	cv::Mat l_block = left_(cv::Range(p_.x - window_size_, p_.x + window_size_), cv::Range(p_.y - window_size_, p_.y + window_size_));
	
	// maximum disparity in which range we search for the actual disparity
	int max_disp = ((p_.y - max_disp_ - window_size_) < 0) ? ( p_.y - window_size_) : max_disp_;
	
	for (size_t disparity = 0; disparity <= max_disp; disparity++) {
		
		// sum of absolute differences
		long sad = 0;

		// right block
		cv::Mat r_block = right_(cv::Range(p_.x - window_size_, p_.x + window_size_), cv::Range(p_.y - disparity - window_size_, p_.y - disparity + window_size_));

		cv::absdiff(l_block, r_block, abs_diff);
		sad = cv::sum(abs_diff)[0];
		
		// searching for disparity where sad is min
		if (sad < min_sad) {
			min_sad = sad;
			min_idx = disparity;
		}
	}
	return min_idx;
}

// Calculating diparity map using naive block-matching algorithm
void naiveDisparityMap(
	const cv::Mat& left_,		// left image
	const cv::Mat& right_,		// right image
	const int& window_size_,	// window size
	const int& max_disp_,		// maximum disparity
	cv::Mat& disp_)				// disparity (output)
{

	disp_ = cv::Mat(left_.rows, left_.cols, CV_8UC1, cv::Scalar(0));
	for (size_t i = window_size_; i < left_.rows - window_size_ - 1; i++) {
		for (size_t j = window_size_; j < left_.cols - window_size_ - 1; j++) {
			int disparity = getMatchingPoint(cv::Point(i, j), left_, right_, window_size_, max_disp_);
			disp_.at<uchar>(i, j) = disparity;
		}
	}
}

// Calculating diparity map using dp algorithm
void dynamicDisparityMap(
	const cv::Mat& left_,		// left image
	const cv::Mat& right_,		// right image
	const int& window_size_,	// window size
	const float& lambda_,		// maximum disparity
	cv::Mat& disp_)				// disparity (output)
{

	// init
	cv::Size size = left_.size();
	cv::Size mat_size = cv::Size(size.width - 2 * window_size_, size.width - 2 * window_size_);
	cv::Mat abs_diff;
	disp_ = cv::Mat(size, CV_8UC1, cv::Scalar(0));

	// calculate C and M for every row
	for (size_t row_idx = window_size_; row_idx < size.height - window_size_; ++row_idx) {


		// init 2.0
		cv::Mat cost = cv::Mat(mat_size, CV_16UC1, cv::Scalar(0));
		cv::Mat match = cv::Mat(mat_size, CV_8UC1, cv::Scalar(0));

		for (size_t cost_row_idx = 1; cost_row_idx < mat_size.height; ++cost_row_idx) {
			cost.at<ushort>(cost_row_idx, 0) = cost_row_idx * lambda_;
			match.at<uchar>(cost_row_idx, 0) = 1;
		}

		for (size_t cost_col_idx = 1; cost_col_idx < mat_size.width; ++cost_col_idx) {
			cost.at<ushort>(0, cost_col_idx) = cost_col_idx * lambda_;
			match.at<uchar>(0, cost_col_idx) = 2;
		}

		// calculating cost and match matrix for current row based on row in left and row in right image
		for (size_t left_col_idx = 1; left_col_idx < mat_size.width; ++left_col_idx) {
			for (size_t right_col_idx = 1; right_col_idx < mat_size.height; ++right_col_idx) {
			
				// left and right region of interests
				cv::Rect l_roi = cv::Rect(left_col_idx, row_idx - window_size_, 2 * window_size_ + 1, 2 * window_size_ + 1);
				cv::Rect r_roi = cv::Rect(right_col_idx, row_idx - window_size_, 2 * window_size_ + 1, 2 * window_size_ + 1);

				// left and right windows
				cv::Mat leftWindow = left_(l_roi);
				cv::Mat rightWindow = right_(r_roi);

				// calculating sad
				cv::absdiff(leftWindow, rightWindow, abs_diff);
				int sad = cv::sum(abs_diff)[0];

				// costs [0] - match, [1] - left occl, [2] - right occl
				int costs[3] = { 0, 0, 0 };
				costs[0] = cost.at<ushort>(right_col_idx - 1, left_col_idx - 1) + sad;
				costs[1] = cost.at<ushort>(right_col_idx - 1, left_col_idx) + lambda_;
				costs[2] = cost.at<ushort>(right_col_idx, left_col_idx - 1) + lambda_;

				// searching for min cost based on cells calculated before
				int min_cost = costs[0], min_idx = 0;
				for (size_t cost_idx = 0; cost_idx < 3; cost_idx++)
				{
					if (costs[cost_idx] < min_cost)
					{
						min_cost = costs[cost_idx];
						min_idx = cost_idx;
					}
				}

				// assigning the cost and match to the minimal cost
				cost.at<ushort>(right_col_idx, left_col_idx) = min_cost;
				match.at<uchar>(right_col_idx, left_col_idx) = min_idx;
			}
		}


		// path planning starting from bottom right
		int disp_row_idx = mat_size.height - 1;
		int disp_col_idx = mat_size.width - 1;

		// iterating over columns
		while (disp_col_idx > 0)
		{
			// match -> step into upper left cell, setting disp
			if (match.at<uchar>(disp_row_idx, disp_col_idx) == 0)
			{
				disp_.at<uchar>(row_idx, disp_col_idx) = abs(disp_row_idx - disp_col_idx);
				disp_row_idx--;
				disp_col_idx--;
			}
			// left occl -> step into upper cell
			else if (match.at<uchar>(disp_row_idx, disp_col_idx) == 1)
			{
				disp_row_idx--;
			}
			// right occl -> step into left cell, setting disp
			else if (match.at<uchar>(disp_row_idx, disp_col_idx) == 2)
			{
				disp_.at<uchar>(row_idx, disp_col_idx) = 0;
				disp_col_idx--;
			}
		}
	}

};

// Calculate 3D X Y Z coordinates from disparity map and calibration parameters
void cameraToWorld(
	const cv::Mat& disp_,		// disparity map
	const float& dmin_,			// additional parameter for shifting
	const float& f_,			// focal length
	const float& b_,			// baseline
	cv::Mat& world_)
{

	// meshgrid for pixel coordinates (u, v)
	cv::Mat xyPlanes[2];
	
	cv::Mat_<Pixel> meshgrid = cv::Mat(disp_.size(), CV_32FC2, cv::Scalar(0));
	meshgrid.forEach([](Pixel& pixel, const int* position) -> void {
		pixel.x = (float)position[0];
		pixel.y = (float)position[1];
	});

	cv::split(meshgrid, xyPlanes);
	cv::Mat u = xyPlanes[0];
	cv::Mat v = xyPlanes[1];

	// shifting to image center from top left corner
	u = u - u.size().height / 2;
	v = v - v.size().width / 2;

	// float and adjusted disparity
	cv::Mat disp_float;
	disp_.convertTo(disp_float, CV_32FC1);
	cv::Mat disp_adjusted = disp_float + dmin_;

	// calc X Y Z coordinates and merge into single matrix 
	cv::Mat X = cv::Mat(disp_.size(), CV_32FC1, cv::Scalar(0));
	cv::Mat Y = cv::Mat(disp_.size(), CV_32FC1, cv::Scalar(0));
	cv::Mat Z = cv::Mat(disp_.size(), CV_32FC1, cv::Scalar(0));

	X = -b_ * (-v * 2 - disp_adjusted) / (2 * disp_adjusted);
	Y = (b_ * u) / disp_adjusted;
	Z = (b_ * f_) / disp_adjusted;

	std::vector<cv::Mat> channels = { X, Y, Z };
	cv::merge(channels, world_);
}