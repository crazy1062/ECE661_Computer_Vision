#include <iostream>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <stack>
#include <opencv.hpp>

using namespace std;
using namespace cv;

#define KAPA 0.04
#define NONMAXWIN 5
#define WINSIZE 5
#define TNONMAXWIN 5
#define TWINSIZE 5
#define NONMAXFLAG 1
#define SOBELWIN 7
#define MAXK 10
#define SCALE 2
#define PI 3.14

struct letterVec{
	letterVec(): width(0), height(0), topleft(Point(0, 0)), rFeature(Mat::zeros(3, 3, CV_32FC1)){};
	vector<Point> blob;
	Point topleft;
	int width;
	int height;
	Mat rFeature;
};

struct cornerVec{
	cornerVec(): s(0), pt(Point(0, 0)){};
	float s;
	Point pt;
};

/*--------------------------------------------------
 * Function: 
 *		Otsu - use Otsu's algorithm to find optimal
 *		threshold
 * Input:
 *		img: input image
 *		lbound: lower bound of the bins
 *		ubound: upper bound of the bins
 * Output:
 *		int thresh
 *--------------------------------------------------
 */

int Otsu(const Mat &img, int lbound, int ubound)
{
	int i, j;
	int numBins = 256;
	float range[] = {0, 256};
	const float* histRange = {range};
	bool uniform = true; bool accumulate = false;
	float width = (float)img.cols;
	float height = (float)img.rows;
	Mat hist;

	// calculate the histogram
	calcHist(&img, 1, 0, Mat(), hist, 1, &numBins, &histRange, uniform, accumulate);

	// calculate the total pixels
	float total_pixels = 0;
	for(i = lbound; i < ubound + 1; i++)
	{
		total_pixels = total_pixels + hist.at<float>(i);
	}

	// calculate the mean level of the original image
	float uT = 0;
	for(i = lbound; i < ubound + 1; i++)
	{
		uT = uT + (float)i*hist.at<float>(i)/total_pixels;
	}

	int thresh = 0;
	float w0, uk;
	float vb;
	float maxvb = 0;
	for(i = lbound; i < ubound + 1; i++)
	{
		uk = 0;
		w0 = 0;
		vb = 0;
		// calculate the mean and probability of occurances for C0
		for(j = lbound; j < i + 1; j++)
		{
			w0 = w0 + hist.at<float>(j)/total_pixels;
			uk = uk + (float)j*hist.at<float>(j)/total_pixels;
		}
		// calculate the between class variance
		vb = w0*(1 - w0)*pow((uT - uk)/(1 - w0) - uk/w0, 2);
		if(vb > maxvb)
		{
			thresh = i;
			maxvb = vb;
		}
	}

	cout << "thresh: " << thresh << endl;

	return thresh;
}


/*--------------------------------------------------
 * Function: 
 *		itr_Otsu - iterative method to use Otsu
 * Input:
 *		img: input image
 *		lbound: lower bound of the bins
 *		ubound: upper bound of the bins
 *		num: the number of iteration
 * Output:
 *		int thresh
 *--------------------------------------------------
 */

int itr_Otsu(const Mat &img, int lbound, int ubound, int num)
{
	int thresh;

	thresh = Otsu(img, lbound, ubound);

	for(int i = 1; i < num; i++)
	{
		thresh = Otsu(img, lbound, thresh);
	}

	return thresh;
}


/*--------------------------------------------------
 * Function: 
 *		getMask - get binary mask of the image after
 *		applying Otsu's algorithm
 * Input:
 *		img: input image
 *		thresh_type: threshold type
 * Output:
 *		Mat mask2
 *--------------------------------------------------
 */

Mat getMask(const Mat& img, int thresh_type)
{
	Mat img_channels[3];
	Mat b_mask, g_mask, r_mask, mask1, mask2;
	split(img, img_channels);

	int thresh_b = itr_Otsu(img_channels[0], 0, 255, 1);
	int thresh_g = itr_Otsu(img_channels[1], 0, 255, 1);
	int thresh_r = itr_Otsu(img_channels[2], 0, 255, 1);

	threshold(img_channels[0], b_mask, thresh_b, 255, thresh_type);
	threshold(img_channels[1], g_mask, thresh_g, 255, thresh_type);
	threshold(img_channels[2], r_mask, thresh_r, 255, thresh_type);

	bitwise_and(b_mask, g_mask, mask1);
	bitwise_and(r_mask, mask1, mask2);

	return mask2;

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
	win.convertTo(win, CV_32FC1);

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
			
			// set interest point to zero if smaller than any other value in the window
			for(int k = 0; k < slideWidth; k++)
			{
				for(int l = 0; l < slideHeight; l++)
				{
					if(corner_s.at<float>(j,i) < slideWin.at<float>(l,k))
						corner_s.at<float>(j,i) = 0;
				}
			}
		}
	}
}

/*-------------------------------------------------
 * Function: 
 *		Harris - Harris corner detection with 
 *		different scales under Gaussion scale 
 *		space
 * Input:
 *		image: original image
 *		scale: scale sigma
 *		win_size: window size of computation of
 *		the corner area
 *		sup_win_size: non maximum suppression area 
 *		window size
 *		haar_win_size: haar filter window size
 * Output:
 *		return Mat corner strength
 *		Mat image
 *-------------------------------------------------
 */

Mat Harris(Mat& image, int sup_win_size, int win_size, int nonMaxFlag = 1)
{
	Mat Ix, Iy;
	Size size = image.size();

	// define the different scale of the scale space
	//cout << "Blurring the image..." << endl;
	GaussianBlur(image, image, Size(win_size,win_size), SCALE, 0);

	Sobel(image, Ix, CV_32FC1, 1, 0, SOBELWIN);
	Sobel(image, Iy, CV_32FC1, 0, 1, SOBELWIN);

	cout << "Calculating the corner strength..." << endl;
	Mat cov(size, CV_32FC3);
	Mat corner_strength = Mat::zeros(size, CV_32FC1);

	Mat temp;

	// store dx2, dy2, dxdy in a 3 channel matrix
	for(int i = 0; i < size.height; i++)
	{
		float* cov_data = (float*)(cov.data + i*cov.step);
		const float* dxdata = (const float*)(Ix.data + i*Ix.step);
		const float* dydata = (const float*)(Iy.data + i*Iy.step);
		for(int j = 0; j < size.width; j++)
		{
			float dx = dxdata[j];
			float dy = dydata[j];

			cov_data[j*3] = dx*dx;
			cov_data[j*3+1] = dx*dy;
			cov_data[j*3+2] = dy*dy;
		}
	}

	// normalization factor
	float norm = (float)1/(win_size*win_size);

	// calculate the corner strength
	for(int i = win_size/2; i < corner_strength.cols - win_size/2; i++)
	{
		for(int j = win_size/2; j < corner_strength.rows - win_size/2; j++)
		{
			temp = getWindow(cov, Point(i,j), win_size);
			float sumdx2 = sum(temp).val[0];
			float sumdxy = sum(temp).val[1];
			float sumdy2 = sum(temp).val[2];
			corner_strength.at<float>(j,i) = (float)norm*(sumdx2*sumdy2 - sumdxy*sumdxy - KAPA*(sumdx2 + sumdy2)*(sumdx2 + sumdy2));
		}
	}

	// Non Maximum Suppression
	if(nonMaxFlag == 1){
		cout << "Non Maximum Suppression Process..." << endl;
		nonMaxSup(corner_strength, sup_win_size);
	}
	cout << "Corner strength processing completed." << endl;

	return corner_strength;
}


/*------------------------------------------------
 * Function: 
 *		floodfill_label - 4-connected component 
 *		labeling algorithm using flood fill
 * Input:
 *		binImg: binary image mask
 *		labelImg: labeled image(CV_32SC1)
 *		blobs: blob containers
 * Output:
 *		Mat labelImg
 *		vector<vector<Point>> blobs
 *------------------------------------------------
 */

void floodfill_label(const Mat &binImg, Mat &labelImg, vector<vector<Point>> &blobs)
{
	labelImg.release();
	binImg.convertTo(labelImg, CV_32SC1);

	int label = 1;
	int rows = binImg.rows - 1;
	int cols = binImg.cols - 1;
	
	for(int i = 1; i < rows - 1; i++)
	{
		int* data = labelImg.ptr<int>(i);
		for(int j = 1; j < cols - 1; j++)
		{
			if(data[j] == 255)
			{
				// create an empty stack to contain any connected points in the blob
				stack<pair<int, int>> neighborPixels;
				// push the seed point of the foreground
				neighborPixels.push(pair<int, int>(i,j));
				++label;
				// initialize blob vector to contain points in a blob
				vector<Point> blob;
				while(!neighborPixels.empty())
				{
					pair<int, int> curPixel = neighborPixels.top();
					int curX = curPixel.first;
					int curY = curPixel.second;
					labelImg.at<int>(curX, curY) = label;

					// store the point int the blob
					blob.push_back(Point(curY, curX));
					// pop the traversed point
					neighborPixels.pop();

					// 4-connected components neighbors searching
					if(curY > 0){
						if(labelImg.at<int>(curX, curY - 1) == 255)
						{
							neighborPixels.push(pair<int, int>(curX, curY - 1));
						}
					}
					if(curY < cols){
						if(labelImg.at<int>(curX, curY + 1) == 255)
						{
							neighborPixels.push(pair<int, int>(curX, curY + 1));
						}
					}
					if(curX > 0){
						if(labelImg.at<int>(curX - 1, curY) == 255)
						{
							neighborPixels.push(pair<int, int>(curX - 1, curY));
						}
					}
					if(curX < rows){
						if(labelImg.at<int>(curX + 1, curY) == 255)
						{
							neighborPixels.push(pair<int, int>(curX + 1, curY));
						}
					}
				}
				// collect blobs
				blobs.push_back(blob);
			}
		}
	}
}


/*------------------------------------------------
 * Function: 
 *		showBlobs - show blobs collected from the  
 *		flood fill algorithm
 * Input:
 *		blobs: blob container
 *		size: the size of the binary mask
 * Output:
 *		Mat output
 *------------------------------------------------
 */

Mat showBlobs(const vector<vector<Point>> &blobs, Size size)
{
	Mat output = Mat::zeros(size, CV_8UC3);

	for(int i = 0; i < blobs.size(); i++)
	{
		unsigned char r = 255*(rand()/(1.0 + RAND_MAX));
		unsigned char g = 255*(rand()/(1.0 + RAND_MAX));
		unsigned char b = 255*(rand()/(1.0 + RAND_MAX));

		for(int j = 0; j < blobs[i].size(); j++)
		{
			int x = blobs[i][j].x;
			int y = blobs[i][j].y;

			output.at<Vec3b>(y, x)[0] = b;
			output.at<Vec3b>(y, x)[1] = g;
			output.at<Vec3b>(y, x)[2] = r;
		}
	}

	return output;
}


/*--------------------------------------------------
 * Function: 
 *		extractLetter - extract the small frames 
 *		including the character blob
 * Input:
 *		blobs: blob container
 *		mask: the binary image mask
 * Output:
 *		vector<letterVec> lvVec
 *--------------------------------------------------
 */

vector<letterVec> extractLetter(const vector<vector<Point>> &blobs, const Mat& mask)
{
	vector<letterVec> lvVec;

	for(int i = 0; i < blobs.size(); i++)
	{
		letterVec lv;
		float xmin = 10e10, ymin = 10e10, xmax = 0, ymax = 0;

		// store the boundary of the blob
		for(int j = 0; j < blobs[i].size(); j++)
		{
			int x = blobs[i][j].x;
			int y = blobs[i][j].y;

			if(x < xmin) xmin = x;
			if(x > xmax) xmax = x;
			if(y < ymin) ymin = y;
			if(y > ymax) ymax = y;
		}

		// discard noise blobs and background blob
		if(blobs[i].size() < 100) continue;
		if(xmin == 0 && ymin == 0) continue;

		int lvwidth = xmax - xmin;
		int lvheight = ymax - ymin;
		// give a size for the frame containing the letter
		Mat rtmp = Mat::zeros(Size(lvwidth + 20, lvheight + 20), CV_8UC1);

		// copy blob to the frame
		for(int k = 0; k < blobs[i].size(); k++)
		{
			int x = blobs[i][k].x;
			int y = blobs[i][k].y;

			rtmp.at<uchar>(y - ymin + 10, x - xmin + 10) = mask.at<uchar>(y, x);
		}

		lv.blob = blobs[i];
		lv.topleft = Point(xmin, ymin);
		lv.width = lvwidth;
		lv.height = lvheight;
		lv.rFeature = rtmp.clone();

		lvVec.push_back(lv);
	}

	return lvVec;
}


/*--------------------------------------------------
 * Function: 
 *		sortByLetter - sort letter function in the 
 *		training image for stl sort algorithm
 * Input:
 *		lhs: left hand side element
 *		rhs: right hand side element
 * Output:
 *		bool
 *--------------------------------------------------
 */


bool sortByLetter(const letterVec &lhs, const letterVec &rhs)
{
	int lx = lhs.topleft.x;
	int ly = lhs.topleft.y;
	int rx = rhs.topleft.x;
	int ry = rhs.topleft.y;

	// not every letter stay on the same row, so the sequence
	// of the alphbet in the letter vector container is not
	// correct, need to be sorted to meet the demand
	if(ry - ly < 10 && ry - ly > -10){
		if(lx <= rx){
			return true;
		}
	}

	return false;
}


/*--------------------------------------------------
 * Function: 
 *		cvtCornerVec - convert corner pixels to 
 *		corner vectors containing point position
 *		and corner strength
 * Input:
 *		corner: corner strength matrix
 * Output:
 *		vector<cornerVec> cvVec
 *--------------------------------------------------
 */


vector<cornerVec> cvtCornerVec(const Mat& corner)
{
	vector<cornerVec> cvVec;
	cornerVec cVec;

	for(int i = 0; i < corner.cols; i++)
	{
		for(int j = 0; j < corner.rows; j++)
		{
			cVec.pt = Point(i, j);
			cVec.s = corner.at<float>(j, i);
			cvVec.push_back(cVec);
		}
	}

	return cvVec;
}

/*--------------------------------------------------
 * Function: 
 *		sortByCorner - sort corner strength function
 *		for the stl sort algorithm
 * Input:
 *		lhs: left hand side element
 *		rhs: right hand side element
 * Output:
 *		bool
 *--------------------------------------------------
 */

bool sortByCorner(const cornerVec &lhs, const cornerVec &rhs)
{
	return lhs.s > rhs.s;
}


/*--------------------------------------------------
 * Function: 
 *		drawMaxk - draw k largest corners in the 
 *		frame
 * Input:
 *		lmat: frame containing the letter
 *		mk: k largest point container
 * Output:
 *		Mat lcmat
 *--------------------------------------------------
 */

Mat drawMaxk(const Mat& lmat, vector<Point> mk)
{
	Mat lcmat = lmat.clone();
	cvtColor(lcmat, lcmat, CV_GRAY2BGR);

	for(int i = 0; i < mk.size(); i++)
	{
		circle(lcmat, mk[i], 2, Scalar(0, 0, 255), -1);
	}

	return lcmat;
}


/*--------------------------------------------------
 * Function: 
 *		findCentroid - find the centroid of the 
 *		blob
 * Input:
 *		lmat: frame containing the letter
 * Output:
 *		Point C
 *--------------------------------------------------
 */


Point findCentroid(const Mat& lmat)
{
	Point C;
	int sumX = 0, sumY = 0;
	int size = 0;

	for(int i = 0; i < lmat.cols; i++)
	{
		for(int j = 0; j < lmat.rows; j++)
		{
			if(lmat.at<uchar>(j, i) > 0)
			{
				sumX = sumX + i;
				sumY = sumY + j;
				size++;
			}
		}
	}

	C.x = sumX/size;
	C.y = sumY/size;

	return C;
}


/*--------------------------------------------------
 * Function: 
 *		extractFeature - extract the angle features
 *		for the letter frame
 * Input:
 *		lomat: original letter frame
 *		index: index for the letter frame
 *		name: name for training image or test image
 *		nonmax_win: non-maximum window size
 *		winsize: corner strength window size
 * Output:
 *		vector<float> angles
 *--------------------------------------------------
 */

vector<float> extractFeature(Mat lomat, int index, string name, int nonmax_win, int winsize)
{
	Mat corner, lcmat;
	Mat lmat = lomat.clone();
	vector<cornerVec> cvVec;
	vector<Point> maxKpt;
	vector<float> vangles, angles;

	// calculate the Harris corner
	corner = Harris(lmat, nonmax_win, winsize);
	// convert corner strength to corner vectors
	cvVec = cvtCornerVec(corner);
	// sort the corner strength descendingly 
	sort(cvVec.begin(), cvVec.end(), sortByCorner);

	// choose k largest corners
	for(int j = 0; j < MAXK; j++)
	{
		maxKpt.push_back(cvVec[j].pt);
	}

	lcmat = drawMaxk(lmat, maxKpt);

	Point C = findCentroid(lomat);
	circle(lcmat, C, 2, Scalar(0, 255, 255), -1);

	// save shape vector drawings
	stringstream ss;
	ss << index;
	string str = ss.str();
	imwrite(name + str + ".jpg", lcmat);

	// calculate the angle between the x axis and the corner vector
	for(int k = 0; k < maxKpt.size(); k++)
	{
		Vec<float, 1> mg, angle;
		Vec<float, 1> vx = (float)(maxKpt[k].x - C.x);
		Vec<float, 1> vy = (float)(C.y - maxKpt[k].y);
		cartToPolar(vx, vy, mg, angle);
		vangles.push_back(angle.val[0]);
	}

	// sort the angle between the x axis and the corner vector
	sort(vangles.begin(), vangles.end());

	// calculate the angles between consecutive corner vectors,
	// the number of angles should be equal to k
	for(int l = 0; l < vangles.size(); l++)
	{
		float angle;
		if(l == vangles.size() - 1){
			angle = 2*PI - vangles[l - 1] + vangles[0];
		}else{
			angle = vangles[l + 1] - vangles[l];
		}
		angles.push_back(angle);
	}

	return angles;
}


/*--------------------------------------------------------
 * Function: 
 *		trainShape - train the shape vector features from
 *		the training image
 * Input:
 *		lvVec: letter vector
 *		letter_name: letter name array
 * Output:
 *		unordered_map<const char*, vector<float>> letters
 *-------------------------------------------------------
 */

unordered_map<const char*, vector<float>> trainShape(const vector<letterVec> &lvVec, const char* letter_name[])
{
	// initialze a hash table to contain training shape vectors
	unordered_map<const char*, vector<float>> letters;
	Mat lmat;
	vector<float> angles;

	// calculate 26 letter shape vectors and insert to the hash table
	for(int i = 0; i < 26; i++)
	{
		lmat = lvVec[i].rFeature;
		
		angles = extractFeature(lmat, i, "train", NONMAXWIN, WINSIZE);

		letters.insert(make_pair(letter_name[i], angles));
		angles.clear();
	}

	return letters;
}


/*--------------------------------------------------
 * Function: 
 *		rotateFeature - rotate the shape vector to 
 *		match the nearest Euclidean distance
 * Input:
 *		angle: test angle container
 *		model: trained angle container
 * Output:
 *		float minDist
 *--------------------------------------------------
 */


float rotateFeature(vector<float> angle, vector<float> model)
{
	float minDist = 10e10, dist;

	for(int i = 0; i < angle.size(); i++)
	{
		dist = 0;
		rotate(angle.begin(), angle.begin()+1, angle.end());
		for(int j = 0; j < model.size(); j++)
		{
			dist = dist + (model[j] - angle[j])*(model[j] - angle[j]);
		}
		if(dist < minDist) minDist = dist;
	}

	return minDist;
}


/*--------------------------------------------------
 * Function: 
 *		recogShape - given trained shape vectors and
 *		test image letter vectors, recognize letters
 * Input:
 *		lvVec: letter vectors container
 *		letters: hash table for the letter vectors
 *		letter_name: letter name array
 *		test: test binary mask image
 * Output:
 *		Mat test
 *--------------------------------------------------
 */

Mat recogShape(vector<letterVec> &lvVec, unordered_map<const char*, vector<float>> letters, const char* letter_name[], Mat& test)
{
	Mat lmat;
	vector<float> angles;
	char* result;

	for(int i = 0; i < lvVec.size(); i++)
	{
		float minDist = 10e10, dist;
		lmat = lvVec[i].rFeature;

		// get the angles as the shape vectors for the test image
		angles = extractFeature(lmat, i, "test", TNONMAXWIN, TWINSIZE);

		for(int j = 0; j < 26; j++)
		{
			// look up hash table to get the angle model
			vector<float> angle_model = letters[letter_name[j]];
			// rotate shape vectors to match the possible correct letter rotation
			dist = rotateFeature(angles, angle_model);

			// find the nearest distance between model vector and the rotated vector
			if(dist < minDist){
				minDist = dist;
				// get the recognized result
				result = (char*)letter_name[j];
			}
		}

		// put recognized result text on the image
		string str = string(result);
		putText(test, str, lvVec[i].topleft, FONT_HERSHEY_SIMPLEX, 1, Scalar(127, 127, 127), 3);
	}

	return test;
}



int main()
{
	char* train_name = "Training.jpg";
	char* test_name1 = "Image1.jpg";
	char* test_name2 = "Image2.jpg";
	char* test_name3 = "Image3.jpg";
	char* test_name4 = "Image4.jpg";
	char* test_name5 = "Image5.jpg";
	char* test_name6 = "Image6.jpg";

	const char* letter_name[26] = {"A", "B", "C", "D", "E", "F", 
	"G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", 
	"S", "T", "U", "V", "W", "X", "Y", "Z"};

	Mat train = imread(train_name, CV_LOAD_IMAGE_COLOR);
	Mat test1 = imread(test_name1, CV_LOAD_IMAGE_COLOR);
	Mat test2 = imread(test_name2, CV_LOAD_IMAGE_COLOR);
	Mat test3 = imread(test_name3, CV_LOAD_IMAGE_COLOR);
	Mat test4 = imread(test_name4, CV_LOAD_IMAGE_COLOR);
	Mat test5 = imread(test_name5, CV_LOAD_IMAGE_COLOR);
	Mat test6 = imread(test_name6, CV_LOAD_IMAGE_COLOR);

	/*-------------------------------------------------
	 * character recognition training process
	 *-------------------------------------------------
	 */

	Mat mask = getMask(train, 0);
	bitwise_not(mask, mask);

	vector<vector<Point>> blobs;

	Mat labelImg;
	// component labeling
	floodfill_label(mask, labelImg, blobs);
	//Mat colorLabelImg;
	//colorLabelImg = showBlobs(blobs, mask.size());
	//imshow("color", colorLabelImg);
	
	// extract training letter blobs and sort letters in the alphabet sequence
	vector<letterVec> lvVec;
	lvVec = extractLetter(blobs, mask);
	sort(lvVec.begin(), lvVec.end(), sortByLetter);

	// construct hash table model and train data
	unordered_map<const char*, vector<float>> letters;
	letters = trainShape(lvVec, letter_name);
	

	/*-------------------------------------------------
	 * character recognition test process
	 *-------------------------------------------------
	 */

	
	Mat mask_test = getMask(test3, 0);
	// for test 6 image, do not use bitwise_not
	bitwise_not(mask_test, mask_test);

	blobs.clear();
	// component labeling
	floodfill_label(mask_test, labelImg, blobs);
	
	lvVec.clear();
	// extract test letter blobs and recognize them
	lvVec = extractLetter(blobs, mask_test);
	Mat mat_result = recogShape(lvVec, letters, letter_name, test3);

	imshow("recognized", mat_result);
	imwrite("recognized.jpg", mat_result);

	waitKey(0);

	return 0;

}