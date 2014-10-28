#include <opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;


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
	int total_pixels = 0;
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
 *		calcTexture - calculate the simplest  
 *		texture of the image
 * Input:
 *		img: original image
 * Output:
 *		return Mat T
 *------------------------------------------------
 */

Mat calcTexture(const Mat& img)
{
	Mat T(img.rows, img.cols, CV_32FC3);
	Mat win3, win5, win7, mdiff;
	Scalar m, m2;

	for(int i = 0; i < img.rows; i++)
	{
		float* T_data = (float*)(T.data + i*T.step);

		for(int j = 0; j < img.cols; j++)
		{
			// get corresponding neighborhood window of the pixel
			win3 = getWindow(img, Point(j, i), 3);
			win5 = getWindow(img, Point(j, i), 5);
			win7 = getWindow(img, Point(j, i), 7);

			// the variance = E[X^2] - [E(X)].^2
			m = mean(win3);
			m2 = mean(win3.mul(win3));
			T_data[j*3] = m2[0] - m[0]*m[0];
			m = mean(win5);
			m2 = mean(win5.mul(win5));
			T_data[j*3 + 1] = m2[0] - m[0]*m[0];
			m = mean(win7);
			m2 = mean(win7.mul(win7));
			T_data[j*3 + 2] = m2[0] - m[0]*m[0];
		}
	}

	return T;
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
 *		calcContour - the function to calculate the
 *		contour of the segmented image
 * Input:
 *		img: the original image
 *		mask: former calculated binary mask
 * Output:
 *		Mat contour
 *--------------------------------------------------
 */

Mat calcContour(const Mat& img, Mat mask)
{
	Mat contour;
	img.copyTo(contour);

	uchar* img_data = (uchar*)contour.data;
	uchar* mask_data = (uchar*)mask.data;

	for(int i = 1; i < img.cols - 1; i++)
	{
		for(int j = 1; j < img.rows - 1; j++)
		{
			// find the edge pixel by 4-connected neighborhood
			if(mask_data[j*mask.step + i] == 0)
			{
				if(!(mask_data[(j - 1)*mask.step + i] == 255 && mask_data[(j + 1)*mask.step + i] == 255 &&
				mask_data[j*mask.step + i - 1] == 255 && mask_data[j*mask.step + i + 1] == 255))
				{
					if(mask_data[(j - 1)*mask.step + i] == 255 || mask_data[(j + 1)*mask.step + i] == 255 ||
					mask_data[j*mask.step + i - 1] == 255 || mask_data[j*mask.step + i + 1] == 255)
					{
						for(int k = 0; k < 3; k++)
						{
							img_data[j*img.step + i*3 + k] = 0;
						}
					}
				}
			}
		}
	}

	return contour;
}


int main()
{

	char* img1_name = "pic1.jpg";
	char* img2_name = "pic2.jpg";
	Mat img1 = imread(img1_name, CV_LOAD_IMAGE_COLOR);
	Mat img2 = imread(img2_name, CV_LOAD_IMAGE_COLOR);
	
	Mat img_channels[3];
	Mat img1_gray, img1_masked, img3;
	Mat img2_gray, img2_masked;
	split(img1, img_channels);
	Mat b_mask, g_mask, r_mask, mask1, mask2;

	/*-------------------------------------
	 * image 1 result (lake)
	 *-------------------------------------
	 */

	// iterative Otsu
	int thresh_b = itr_Otsu(img_channels[0], 0, 255, 1);
	int thresh_g = itr_Otsu(img_channels[1], 0, 255, 2);
	int thresh_r = itr_Otsu(img_channels[2], 0, 255, 2);

	// apply threshold value to get binary masks
	threshold(img_channels[0], b_mask, thresh_b, 255, 0);
	threshold(img_channels[1], g_mask, thresh_g, 255, 0);
	threshold(img_channels[2], r_mask, thresh_r, 255, 0);
	// get the final binary mask
	bitwise_and(b_mask, g_mask, mask1);
	bitwise_and(r_mask, mask1, mask2);
	// get contours
	Mat img1_contour = calcContour(img1, mask2);
	imshow("img1_contour", img1_contour);
	imwrite("img1_Otsu_contour.jpg", img1_contour);
	// segment image out
	cvtColor(mask2, mask2, CV_GRAY2BGR);
	bitwise_and(img1, mask2, img1_masked);
	
	Mat T_channels[3];
	Mat maskT3, maskT5, maskT7;
	Mat img1_T, img2_T, mask3, mask4;
	//img1_gray = 0.0721*img_channels[0] + 0.7154*img_channels[1] + 0.2125*img_channels[2];
	cvtColor(img1, img1_gray, CV_BGR2GRAY);
	//threshold(img1_gray, img3, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	imshow("img1_gray", img1_gray);
	
	// calculate the texture
	Mat T = calcTexture(img1_gray);

	split(T, T_channels);
	Mat T3_norm, T5_norm, T7_norm;
	
	// normalize the texture features
	normalize(T_channels[0], T3_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	normalize(T_channels[1], T5_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	normalize(T_channels[2], T7_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	T3_norm.convertTo(T3_norm, CV_8U, 255.0);
	T5_norm.convertTo(T5_norm, CV_8U, 255.0);
	T7_norm.convertTo(T7_norm, CV_8U, 255.0);

	imshow("img1_T3_norm", T3_norm);
	imwrite("img1_T3_norm.jpg", T3_norm);
	imshow("img1_T5_norm", T5_norm);
	imwrite("img1_T5_norm.jpg", T5_norm);
	imshow("img1_T7_norm", T7_norm);
	imwrite("img1_T7_norm.jpg", T7_norm);

	// apply Otsu as we did in RGB
	int thresh1 = itr_Otsu(T3_norm, 0, 255, 1);
	int thresh2 = itr_Otsu(T5_norm, 0, 255, 1);
	int thresh3 = itr_Otsu(T7_norm, 0, 255, 1);

	// apply threshold value to get binary masks
	threshold(T3_norm, maskT3, thresh1, 255, 1);
	threshold(T5_norm, maskT5, thresh2, 255, 1);
	threshold(T7_norm, maskT7, thresh3, 255, 1);
	// get the final binary mask
	bitwise_and(maskT3, maskT5, mask3);
	bitwise_and(maskT7, mask3, mask4);
	mask4.convertTo(mask4, img1.type());
	// get contours
	Mat img1T_contour = calcContour(img1, mask4);
	imshow("img1T_contour", img1T_contour);
	imwrite("img1_texture_contour.jpg", img1T_contour);

	cvtColor(mask4, mask4, CV_GRAY2BGR);
	bitwise_and(img1, mask4, img1_T);

	imshow("img1_b", b_mask);
	imwrite("img1_bmask.jpg", b_mask);
	imshow("img1_g", g_mask);
	imwrite("img1_gmask.jpg", g_mask);
	imshow("img1_r", r_mask);
	imwrite("img1_rmask.jpg", r_mask);
	imshow("img1_maskT3", maskT3);
	imwrite("img1_T3mask.jpg", maskT3);
	imshow("img1_maskT5", maskT5);
	imwrite("img1_T5mask.jpg", maskT5);
	imshow("img1_maskT7", maskT7);
	imwrite("img1_T7mask.jpg", maskT7);
	imshow("img1_mask2", mask2);
	imwrite("img1_mask2.jpg", mask2);
	imshow("img1_mask4", mask4);
	imwrite("img1_mask4.jpg", mask4);
	imshow("img1_T_seg", img1_T);
	imwrite("img1_T.jpg", img1_T);
	imshow("img1", img1_masked);
	imwrite("img1_seg.jpg", img1_masked);
	
	
	/*-------------------------------------
	 * image 2 result (tiger)
	 *-------------------------------------
	 */

	
	split(img2, img_channels);

	// iterative Otsu
	thresh_b = itr_Otsu(img_channels[0], 0, 255, 2);
	thresh_g = itr_Otsu(img_channels[1], 0, 255, 2);
	thresh_r = itr_Otsu(img_channels[2], 0, 255, 2);

	// apply threshold value to get binary masks
	threshold(img_channels[0], b_mask, thresh_b, 255, 0);
	threshold(img_channels[1], g_mask, thresh_g, 255, 0);
	threshold(img_channels[2], r_mask, thresh_r, 255, 0);
	// get the final binary mask
	bitwise_and(b_mask, g_mask, mask1);
	bitwise_and(r_mask, mask1, mask2);
	// get contours
	Mat img2_contour = calcContour(img2, mask2);
	imshow("img2_contour", img2_contour);
	imwrite("img2_Otsu_contour.jpg", img2_contour);

	cvtColor(mask2, mask2, CV_GRAY2BGR);
	bitwise_and(img2, mask2, img2_masked);

	cvtColor(img2, img2_gray, CV_BGR2GRAY);

	imshow("img2_gray", img2_gray);

	// calculate the texture
	T = calcTexture(img2_gray);

	split(T, T_channels);

	// normalize the texture features
	normalize(T_channels[0], T3_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	normalize(T_channels[1], T5_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	normalize(T_channels[2], T7_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	T3_norm.convertTo(T3_norm, CV_8U, 255.0);
	T5_norm.convertTo(T5_norm, CV_8U, 255.0);
	T7_norm.convertTo(T7_norm, CV_8U, 255.0);

	imshow("img2_T3_norm", T3_norm);
	imwrite("img2_T3_norm.jpg", T3_norm);
	imshow("img2_T5_norm", T5_norm);
	imwrite("img2_T5_norm.jpg", T5_norm);
	imshow("img2_T7_norm", T7_norm);
	imwrite("img2_T7_norm.jpg", T7_norm);

	// apply Otsu as we did in RGB
	thresh1 = itr_Otsu(T3_norm, 0, 255, 1);
	thresh2 = itr_Otsu(T5_norm, 0, 255, 1);
	thresh3 = itr_Otsu(T7_norm, 0, 255, 1);

	// apply threshold value to get binary masks
	threshold(T3_norm, maskT3, thresh1, 255, 0);
	threshold(T5_norm, maskT5, thresh2, 255, 0);
	threshold(T7_norm, maskT7, thresh3, 255, 0);
	// get the final binary mask
	bitwise_and(maskT3, maskT5, mask3);
	bitwise_and(maskT7, mask3, mask4);
	mask4.convertTo(mask4, img2.type());
	// get contours
	Mat img2T_contour = calcContour(img2, mask4);
	imshow("img2T_contour", img2T_contour);
	imwrite("img2_texture_contour.jpg", img2T_contour);

	cvtColor(mask4, mask4, CV_GRAY2BGR);
	bitwise_and(img2, mask4, img2_T);

	imshow("img2_b", b_mask);
	imwrite("img2_bmask.jpg", b_mask);
	imshow("img2_g", g_mask);
	imwrite("img2_gmask.jpg", g_mask);
	imshow("img2_r", r_mask);
	imwrite("img2_rmask.jpg", r_mask);
	imshow("img2_maskT3", maskT3);
	imwrite("img2_T3mask.jpg", maskT3);
	imshow("img2_maskT5", maskT5);
	imwrite("img2_T5mask.jpg", maskT5);
	imshow("img2_maskT7", maskT7);
	imwrite("img2_T7mask.jpg", maskT7);
	imshow("img2_mask2", mask2);
	imwrite("img2_mask2.jpg", mask2);
	imshow("img2_mask4", mask4);
	imwrite("img2_mask4.jpg", mask4);
	imshow("img2_T_seg", img2_T);
	imwrite("img2_T.jpg", img2_T);
	imshow("img2", img2_masked);
	imwrite("img2_seg.jpg", img2_masked);
	

	waitKey(0);

	return 0;

}
