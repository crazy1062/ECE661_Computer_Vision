#include <opencv.hpp>
#include <iostream>
#include <vector>
#include <iterator>
#include <opencv2\nonfree\nonfree.hpp>

using namespace std;
using namespace cv;

#define KAPA 0.04
#define MAXPT 100
#define SCALE 1
#define HARRISTHRESH 2e+12
#define HARRISTHRESH2 4e+12
#define HAARWIN 6
#define HARRISWIN 5
#define NONMAXWIN 15
#define NCCTHRESH 0.5
#define NCCRATIO 0.7
#define NCCWIN 41
#define SSDTHRESH 10000
#define SSDRATIO 0.7
#define SSDWIN 43

struct descriptor{
	int index;
	Point pt[MAXPT];
	Mat win[MAXPT];
};

struct pointPair{
	Point p1;
	Point p2;
};

Mat getWindow(const Mat& img, Point pt, int win_size);
void nonMaxSup(Mat& corner_s, int win_size);
Mat scaleHarris(Mat& image, double scale, int win_size, int sup_win_size, int haar_win_size);
void featureExtract(descriptor& dcrip, const Mat& corner_s, double thresh, const Mat& image, int win_size);
void NCC(const descriptor& dcrip1, const descriptor& dcrip2, double thresh, double ratio_thresh, vector<pointPair>& pair, int win_size);
void SSD(const descriptor& dcrip1, const descriptor& dcrip2, double thresh, double ratio_thresh, vector<pointPair>& pair, int win_size);
void drawPairs(const Mat& img1, const Mat& img2, Mat& imgOut, vector<pointPair>& pair);
void drawPoints(Mat& img, const Mat& corner_s, double thresh, double scale, int win_size);


/*-------------------------------------------------
 * Function: 
 *		scaleHarris - Harris corner detection with 
 *		different scales under Gaussion scale 
 *		space
 * Input:
 *		image: original image
 *		scale: scale sigma
 *		win_size: window size of computation of
 *		the corner area
 *		sup_win_size: non maximum suppression area 
 *		window size
 * Output:
 *		return Mat corner strength
 *-------------------------------------------------
 */

Mat scaleHarris(Mat& image, double scale, int win_size, int sup_win_size, int haar_win_size)
{
	Mat Ix, Iy, Ix2, Iy2, Ixy;
	Size size = image.size();
	int ddepth = -1;

	GaussianBlur(image, image, Size(win_size,win_size), scale, 0);

	Mat tmpknlp = Mat::ones(win_size+1, (win_size+1)/2, CV_64FC1);
	Mat tmpknln = -1*tmpknlp;

	Mat kernelx, kernely;
	hconcat(tmpknln, tmpknlp, kernelx);
	vconcat(tmpknlp.t(), tmpknln.t(), kernely);

	cout << "kernelx" << kernelx << endl;
	cout << "kernely" << kernely << endl;

	filter2D(image, Ix, CV_64FC1, kernelx);
	filter2D(image, Iy, CV_64FC1, kernely);
	//Sobel(image, Ix, CV_64FC1, 1, 0, 3);
	//Sobel(image, Iy, CV_64FC1, 0, 1, 3);
	//Scharr(image, Ix, CV_64FC1, 1, 0);
	//Scharr(image, Iy, CV_64FC1, 0, 1);
	

	Mat cov(size, CV_64FC3);
	Mat corner_strength(size, CV_64FC1);
	Mat corner_strength2(size, CV_64FC1);
	/*
	Ix2 = Ix.mul(Ix);
	Iy2 = Iy.mul(Iy);
	Ixy = Ix.mul(Iy);

	//GaussianBlur(Ix2, Ix2, Size(win_size,win_size), scale, 0);
	//GaussianBlur(Iy2, Iy2, Size(win_size,win_size), scale, 0);
	//GaussianBlur(Ixy, Ixy, Size(win_size,win_size), scale, 0);
	boxFilter(Ix2, Ix2, -1, Size(win_size, win_size));
	boxFilter(Iy2, Iy2, -1, Size(win_size, win_size));
	boxFilter(Ixy, Ixy, -1, Size(win_size, win_size));

	Mat Ixy_sum = Ix2 + Iy2;

	corner_strength = (Ix2.mul(Iy2) - Ixy.mul(Ixy)) - KAPA*Ixy_sum.mul(Ixy_sum);
	GaussianBlur(corner_strength, corner_strength, Size(win_size, win_size), scale, 0);
	imshow("corner_strength: ", corner_strength);
	
	Mat temp;

	for(int i = 0; i < corner_strength.cols; i++)
	{
		for(int j = 0; j < corner_strength.rows; j++)
		{
			temp = getWindow(corner_strength, Point(i,j), win_size);
			corner_strength2.at<double>(j,i) = (double)sum(temp).val[0]/win_size;
		}
	}
	*/

	Mat temp;

	for(int i = 0; i < size.height; i++)
	{
		double* cov_data = (double*)(cov.data + i*cov.step);
		const double* dxdata = (const double*)(Ix.data + i*Ix.step);
		const double* dydata = (const double*)(Iy.data + i*Iy.step);
		for(int j = 0; j < size.width; j++)
		{
			double dx = dxdata[j];
			double dy = dydata[j];

			cov_data[j*3] = dx*dx;
			cov_data[j*3+1] = dx*dy;
			cov_data[j*3+2] = dy*dy;
		}
	}

	double norm = (double)1/(win_size*win_size);
	cout << "norm: " << norm << endl;

	for(int i = 0; i < corner_strength.cols; i++)
	{
		for(int j = 0; j < corner_strength.rows; j++)
		{
			temp = getWindow(cov, Point(i,j), win_size);
			double sumdx2 = sum(temp).val[0];
			double sumdxy = sum(temp).val[1];
			double sumdy2 = sum(temp).val[2];
			corner_strength2.at<double>(j,i) = (double)norm*(sumdx2*sumdy2 - sumdxy*sumdxy - KAPA*(sumdx2 + sumdy2)*(sumdx2 + sumdy2));
		}
	}

	nonMaxSup(corner_strength2, sup_win_size);

	return corner_strength2;
}

/*------------------------------------------------
 * Function: 
 *		getWindow - get the region of interest for 
 *		the interest point
 * Input:
 *		img: original image
 *		pt: interest point
 *		win_size: window size of the region around
 *		the interest point
 * Output:
 *		return Mat window
 *------------------------------------------------
 */

Mat getWindow(const Mat& img, Point pt, int win_size)
{
	Mat win;
	Size img_size = img.size();
	int img_x = img_size.width;
	int img_y = img_size.height;

	if(pt.x < 0 || pt. x > img_size.width){
		cout << "GETWINDOW expects x to be within width of the image";
		return win;
	}
	if(pt.y < 0 || pt.y > img_size.height){
		cout << "GETWINDOW expects y to be within height of the image";
		return win;
	}

	int x_start = pt.x - win_size/2;
	if(x_start < 0) x_start = 0;

	int x_end = pt.x + win_size/2 + 1;
	if(x_end > img_x) x_end = img_x;

	int y_start = pt.y - win_size/2;
	if(y_start < 0) y_start = 0;

	int y_end = pt.y + win_size/2 + 1;
	if(y_end > img_y) y_end = img_y;

	win = img(Range(y_start, y_end), Range(x_start, x_end));
	win.convertTo(win, CV_64FC1);

	return win;
}

/*------------------------------------------------
 * Function: 
 *		nonMaxSup - non maximum suppression of the  
 *		corner strength
 * Input:
 *		corner_s: corner strength matrix
 *		win_size: non maximum suppression area
 *		window size
 * Output:
 *		Mat corner_s
 *------------------------------------------------
 */

void nonMaxSup(Mat& corner_s, int win_size)
{
	Size img_size = corner_s.size();
	int width = img_size.width;
	int height = img_size.height;

	for(int i = 0; i < width; ++i)
	{
		for(int j = 0; j < height; ++j)
		{
			Point p = Point(i,j);
			Mat slideWin = getWindow(corner_s, p, win_size);
			Size slideWinSize = slideWin.size();
			int slideWidth = slideWinSize.width;
			int slideHeight = slideWinSize.height;
			
			for(int k = 0; k < slideWidth; k++)
			{
				for(int l = 0; l < slideHeight; l++)
				{
					if(corner_s.at<double>(j,i) < slideWin.at<double>(l,k))
						corner_s.at<double>(j,i) = 0;
				}
			}
		}
	}
}

/*------------------------------------------------
 * Function: 
 *		featureExtract - extract the feature of   
 *		the corner point
 * Input:
 *		dcrip: descriptor of the corner point
 *		corner_s: corner strength matrix
 *		thresh: harris corner threshold
 *		image: original image
 *		win_size: window size of the matching
 *		method, could be NCCWIN or SSDWIN
 * Output:
 *		descriptor dcrip
 *------------------------------------------------
 */

void featureExtract(descriptor& dcrip, const Mat& corner_s, double thresh, const Mat& image, int win_size)
{
	int count = 0;
	Size size = corner_s.size();
	int width = size.width;
	int height = size.height;
	int i, j;
	for(i = 0; i < width; i++)
	{
		for(j = 0; j < height; j++)
		{
			if(corner_s.at<double>(j,i) > thresh)
			{
				if(count > MAXPT - 1) break;
				dcrip.index = count++;
				dcrip.pt[dcrip.index].x = i;
				dcrip.pt[dcrip.index].y = j;
				dcrip.win[dcrip.index] = getWindow(image, Point(i,j), win_size);
			}
		}
	}
	cout << "number of corners: " << count << endl;
}

/*------------------------------------------------
 * Function: 
 *		NCC - Normalized Cross Correlation method   
 *		to establish correspondences
 * Input:
 *		dcrip1: the descriptor of the 1st image
 *		dcrip2: the descriptor of the 2nd image
 *		thresh: NCC method threshold - NCCTHRESH
 *		ratio_thresh: NCC ratio threshold -
 *		NCCRATIO
 *		pair: container of pointPair to store
 *		matching pairs
 *		win_size: NCC window size
 * Output:
 *		vector<pointPair> pair
 *------------------------------------------------
 */

void NCC(const descriptor& dcrip1, const descriptor& dcrip2, double thresh, double ratio_thresh, vector<pointPair>& pair, int win_size)
{ 
	Scalar avg1, avg2;
	int i, j;
	double NCCval, ratio;
	Size size = Size(win_size, win_size);
	Mat avg1Mat(size, CV_64FC1), avg2Mat(size, CV_64FC1), C1(size, CV_64FC1), C2(size, CV_64FC1), C(size, CV_64FC1);
	cout << "size: " << size << endl;
	for(i = 0; i < dcrip1.index; i++)
	{
		double maxVal = 0, secondMax = 0;
		pointPair tmpPair = {Point(0,0), Point(0,0)};
		//cout << "dcrip1 size: " << dcrip1.win[i].size() << endl;
		if(dcrip1.win[i].size() != size) continue;
		for(j = 0; j < dcrip2.index; j++)
		{
			if(dcrip2.win[j].size() != size) continue;

			//cout << "dcrip1.win[i]: " << dcrip1.win[i] << endl;
			//cout << "dcrip2.win[j]: " << dcrip2.win[j] << endl;
			avg1 = mean(dcrip1.win[i]);
			avg2 = mean(dcrip2.win[j]);
			//cout << "avg1: " << avg1 << endl;
			//cout << "avg2: " << avg2 << endl;
			subtract(dcrip1.win[i], avg1, avg1Mat);
			subtract(dcrip2.win[j], avg2, avg2Mat);
			//cout << "avg1Mat: " << avg1Mat << endl;
			//cout << "avg2Mat: " << avg2Mat << endl;
			C = avg1Mat.mul(avg2Mat);
			C1 = avg1Mat.mul(avg1Mat);
			C2 = avg2Mat.mul(avg2Mat);
			//cout << "C: " << C << endl;
			//cout << "C1: " << C1 << endl;
			//cout << "C2: " << C2 << endl;

			NCCval = sum(C).val[0]/sqrt(sum(C1).val[0]*sum(C2).val[0]);

			//cout << "sum(C).val[0]: " << sum(C).val[0] << endl;

			if(NCCval > thresh)
			{
				if(NCCval > maxVal){
					secondMax = maxVal;
					maxVal = NCCval;
					tmpPair.p1 = dcrip1.pt[i];
					tmpPair.p2 = dcrip2.pt[j];
				}
			}
		}
		//cout << maxVal << endl;
		ratio = secondMax/maxVal;
		if(ratio < ratio_thresh )
		{
			pair.push_back(tmpPair);
		}
	}
}

/*------------------------------------------------
 * Function: 
 *		SSD - Sum of Squared Differences method    
 *		to establish correspondences
 * Input:
 *		dcrip1: the descriptor of the 1st image
 *		dcrip2: the descriptor of the 2nd image
 *		thresh: SSD method threshold - SSDTHRESH
 *		ratio_thresh: SSD ratio threshold -
 *		SSDRATIO
 *		pair: container of pointPair to store
 *		matching pairs
 *		win_size: SSD window size		
 * Output:
 *		vector<pointPair> pair
 *------------------------------------------------
 */

void SSD(const descriptor& dcrip1, const descriptor& dcrip2, double thresh, double ratio_thresh, vector<pointPair>& pair, int win_size)
{
	int i, j;
	Size size = Size(win_size, win_size);
	Mat I(size, CV_64FC1), I2(size, CV_64FC1);
	cout << "size: " << size << endl;
	double SSDval, ratio;
	for(i = 0; i < dcrip1.index; i++)
	{
		double minVal = 1e+10, secondMin = 1e+10;
		pointPair tmpPair = { Point(0,0), Point(0,0)};
		if(dcrip1.win[i].size() != size) continue;
		for(j = 0; j < dcrip2.index; j++)
		{
			if(dcrip2.win[j].size() != size) continue;

			//cout << "dcrip1.win[i]: " << dcrip1.win[i] << endl;
			//cout << "dcrip2.win[j]: " << dcrip2.win[j] << endl;
			subtract(dcrip1.win[i], dcrip2.win[j], I);
			//cout << "I: " << I << endl;
			I2 = I.mul(I);
			//cout << "I2: " << I2 << endl;
			SSDval = (double)sum(I2).val[0]/(win_size*win_size);
			//cout << "sum(I2): " << sum(I2).val[0] << endl;
			//cout << "SSDval: " << SSDval << endl;

			if(SSDval < thresh)
			{
				if(SSDval < minVal){
					secondMin = minVal;
					minVal = SSDval;
					tmpPair.p1 = dcrip1.pt[i];
					tmpPair.p2 = dcrip2.pt[j];
				}
			}
		}
		//cout << "minVal: " << minVal << endl;
		//cout << "secondMin: " << secondMin << endl;

		ratio = minVal/secondMin;
		cout << "ratio: " << ratio << endl;
		if(ratio < ratio_thresh )
		{
			pair.push_back(tmpPair);
		}
	}
}


/*------------------------------------------------
 * Function: 
 *		drawPairs - draw corners in each image and    
 *		draw corresponding line pairs for two
 * Input:
 *		img1: the first image with points on it
 *		img2: the second image with points on it
 *		imgOut: the final merged image with 
 *		matching lines on it
 *		pair: container of pointPair
 * Output:
 *		Mat imgOut
 *------------------------------------------------
 */

void drawPairs(const Mat& img1, const Mat& img2, Mat& imgOut, vector<pointPair>& pair)
{
	Size size = img1.size();
	img1.copyTo(imgOut(Rect(0, 0, size.width, size.height)));
	img2.copyTo(imgOut(Rect(size.width, 0, size.width, size.height)));
	for(vector<pointPair>::iterator it = pair.begin(); it != pair.end(); ++it) 
	{
		line(imgOut, it->p1, Point(it->p2.x + size.width, it->p2.y), Scalar(255, 0, 0));
	}
}

/*--------------------------------------------------
 * Function: 
 *		drawPoints - draw detected corners no matter    
 *		whether they are chosen for matching
 * Input:
 *		img: the target original image
 *		corner_s: corner strength matrix
 *		thresh: harris corner strength
 *		scale: scale sigma
 *		win_size: harris corner window size
 * Output:
 *		Mat img
 *--------------------------------------------------
 */

void drawPoints(Mat& img, const Mat& corner_s, double thresh, double scale, int win_size)
{
	//GaussianBlur(img, img, Size(win_size,win_size), scale, 0);

	for(int j = 0; j < corner_s.rows; j++)
	{
		for(int i = 0; i < corner_s.cols; i++)
		{
			if(corner_s.at<double>(j,i) > thresh)
			{
				circle(img, Point(i,j), 5, Scalar(0), 2, 8, 0);
			}
		}
	}
}


int main(){

	//char* img_name1 = "pic1.jpg";
	//char* img_name2 = "pic2.jpg";
	char* img_name1 = "pic6.jpg";
	char* img_name2 = "pic7.jpg";

	Mat src, src_gray;
	Mat img1, img2;

	int ddepth = -1;

	img1 = imread(img_name1, CV_LOAD_IMAGE_GRAYSCALE);
	img2 = imread(img_name2, CV_LOAD_IMAGE_GRAYSCALE);

	Size img_size = img1.size();
	
	

	Point2d p(10,0);
	
	Mat corner_s1 = scaleHarris(img1, SCALE, HARRISWIN, NONMAXWIN, HAARWIN);
	Mat corner_s2 = scaleHarris(img2, SCALE, HARRISWIN, NONMAXWIN, HAARWIN);

	imshow("img1", img1);
	imshow("img2", img2);

	descriptor dcrip1;
	descriptor dcrip2;

	vector<pointPair> pair;
	vector<pointPair> pair2;

	//featureExtract(dcrip1, corner_s1, HARRISTHRESH, img1, NCCWIN);
	//featureExtract(dcrip2, corner_s2, HARRISTHRESH2, img2, NCCWIN);
	//NCC(dcrip1, dcrip2, NCCTHRESH, NCCRATIO, pair, NCCWIN);
	featureExtract(dcrip1, corner_s1, HARRISTHRESH, img1, SSDWIN);
	featureExtract(dcrip2, corner_s2, HARRISTHRESH2, img2, SSDWIN);
	SSD(dcrip1, dcrip2, SSDTHRESH, SSDRATIO, pair2, SSDWIN);

	cout << "pair size: " << pair.size() << endl;
	cout << "pair2 size: " << pair2.size() << endl;

	drawPoints(img1, corner_s1, HARRISTHRESH, SCALE, HARRISWIN);
	
	drawPoints(img2, corner_s2, HARRISTHRESH2, SCALE, HARRISWIN);

	namedWindow( "corners_window1", CV_WINDOW_AUTOSIZE );
	imshow( "corners_window1", img1 );
	namedWindow( "corners_window2", CV_WINDOW_AUTOSIZE );
	imshow( "corners_window2", img2 );

	//Mat dst(Size(img_size.width*2, img_size.height), img1.type());
	Mat dst2(Size(img_size.width*2, img_size.height), img1.type());

	//drawPairs(img1, img2, dst, pair);
	drawPairs(img1, img2, dst2, pair2);

	//namedWindow("corner pairs NCC", CV_WINDOW_AUTOSIZE);
	//imshow("corner pairs NCC", dst);
	namedWindow("corner pairs SSD", CV_WINDOW_AUTOSIZE);
	imshow("corner pairs SSD", dst2);

	

	/*-----------------------------------------------------------
	 * SURF corner detection
	 *-----------------------------------------------------------
	 */

	/*
	Mat img_object = imread( img_name1, CV_LOAD_IMAGE_GRAYSCALE );
	Mat img_scene = imread( img_name2, CV_LOAD_IMAGE_GRAYSCALE );

	if( !img_object.data || !img_scene.data )
	{ std::cout<< " --(!) Error reading images " << std::endl; return -1; }

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;

	SurfFeatureDetector detector( minHessian );

	vector<KeyPoint> keypoints_object, keypoints_scene;

	detector.detect( img_object, keypoints_object );
	detector.detect( img_scene, keypoints_scene );

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;

	Mat descriptors_object, descriptors_scene;

	extractor.compute( img_object, keypoints_object, descriptors_object );
	extractor.compute( img_scene, keypoints_scene, descriptors_scene );

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	vector< DMatch > matches;
	matcher.match( descriptors_object, descriptors_scene, matches );

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < descriptors_object.rows; i++ )
	{ double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist );
	printf("-- Min dist : %f \n", min_dist );

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	vector< DMatch > good_matches;

	for( int i = 0; i < descriptors_object.rows; i++ )
	{ if( matches[i].distance < 3*min_dist )
		{ good_matches.push_back( matches[i]); }
	}

	Mat img_matches;
	drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	imshow( "Good Matches", img_matches );
	*/

	waitKey(0);

	return 0;

}