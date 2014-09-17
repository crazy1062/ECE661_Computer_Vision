#include <opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

/*------------------------------------------------
 * Function: 
 *		TDrawChosenLine - 2 steps method to draw
 *		chosen lines on the original image given
 *		several points
 * Input:
 *		
 * Output:
 *		
 *------------------------------------------------
 */

void TDrawChosenLine(Mat& image, const Mat pt1, const Mat pt2,
	const Mat pt3,const Mat pt4, const Mat pt5,
	const Mat pt6,const Mat pt7, const Mat pt8)
{
	Point2d tpt1(pt1.at<double>(0), pt1.at<double>(1));
	Point2d tpt2(pt2.at<double>(0), pt2.at<double>(1));
	Point2d tpt3(pt3.at<double>(0), pt3.at<double>(1));
	Point2d tpt4(pt4.at<double>(0), pt4.at<double>(1));
	Point2d tpt5(pt5.at<double>(0), pt5.at<double>(1));
	Point2d tpt6(pt6.at<double>(0), pt6.at<double>(1));
	Point2d tpt7(pt7.at<double>(0), pt7.at<double>(1));
	Point2d tpt8(pt8.at<double>(0), pt8.at<double>(1));

	line(image, tpt1, tpt2, Scalar(0, 0, 255), 2);
	line(image, tpt3, tpt4, Scalar(0, 0, 255), 2);
	line(image, tpt1, tpt3, Scalar(255, 0, 0), 2);
	line(image, tpt2, tpt4, Scalar(255, 0, 0), 2);
	line(image, tpt5, tpt6, Scalar(0, 255, 0), 2);
	line(image, tpt7, tpt8, Scalar(0, 255, 0), 2);
}

/*------------------------------------------------
 * Function: 
 *		convert2Homogeneous - convert point to 
 *		homogeneous representation
 * Input:
 *		
 * Output:
 *		
 *------------------------------------------------
 */

Mat convert2Homogeneous(const Point2d pt)
{
	Mat h = Mat(pt);
	h.push_back(1.0);
	return h;
}

/*-------------------------------------------------
 * Function: 
 *		normHomogeneous - normalize the homogeneous 
 *		representation
 * Input:
 *		
 * Output:
 *		
 *-------------------------------------------------
 */

void normHomogeneous(Mat& homo)
{
	homo.at<double>(0,0) = homo.at<double>(0,0)/homo.at<double>(2,0);
	homo.at<double>(1,0) = homo.at<double>(1,0)/homo.at<double>(2,0);
	homo.at<double>(2,0) = 1.0;
}

/*------------------------------------------------
 * Function: 
 *		constructLine - cross product of two 
 *		homogeneous coordinates to form a line
 * Input:
 *		
 * Output:
 *		
 *------------------------------------------------
 */

Mat constructLine(const Mat pt1, const Mat pt2)
{
	Mat l = pt1.cross(pt2);
	normHomogeneous(l);
	return l;
}

/*------------------------------------------------
 * Function: 
 *		constructPoint - cross product of two 
 *		homogeneous lines to form a intersection
 * Input:
 *		
 * Output:
 *		
 *------------------------------------------------
 */

Mat constructPoint(const Mat l1, const Mat l2)
{
	Mat pt = l1.cross(l2);
	normHomogeneous(pt);
	return pt;
}

/*------------------------------------------------------------------------
 * Function: 
 *		 imageBackProj - back project the current image to the world plane
 * Input: 
 *		 Mat iFrame - the current frame matrix 
 *		 Mat oFrame - the world plane matrix after warp
 *		 Size frame_size - the size of the input frame matrix
 *		 Mat hVector - H transformation matrix
 * Ouput:
 *		 oFrame
 *------------------------------------------------------------------------
 */


Mat imageBackProj(const Mat& iFrame, Mat& oFrame, const Size frame_size, const Mat hVector)
{
	double xmin, xmax, ymin, ymax, xtmp, ytmp;
	double scale, aspect_ratio;
	int height, width;
	Point2d tmpPoint;
	Mat tmpResult, tmpMat;
	Mat hinv = hVector.inv();

	Mat bound_iFrame(3, 4, CV_64F);
	Mat bound_oFrame(3, 4, CV_64F);

	// construct four points of input frame
	bound_iFrame.at<double>(0,0) = 0;
	bound_iFrame.at<double>(1,0) = 0;
	bound_iFrame.at<double>(2,0) = 1.0;
	bound_iFrame.at<double>(0,1) = (double)(frame_size.width - 1);
	bound_iFrame.at<double>(1,1) = 0;
	bound_iFrame.at<double>(2,1) = 1.0;
	bound_iFrame.at<double>(0,2) = (double)(frame_size.width - 1);
	bound_iFrame.at<double>(1,2) = (double)(frame_size.height - 1);
	bound_iFrame.at<double>(2,2) = 1.0;
	bound_iFrame.at<double>(0,3) = 0;
	bound_iFrame.at<double>(1,3) = (double)(frame_size.height - 1);
	bound_iFrame.at<double>(2,3) = 1.0;
	
	bound_oFrame = hVector*bound_iFrame;

	// calculate the bound values of the projected vertices
	xmin = ymin = 1e10; xmax = ymax = 0;
	for(int i = 0; i < 4; ++i)
	{
		xtmp = bound_oFrame.at<double>(0, i)/bound_oFrame.at<double>(2, i);
		ytmp = bound_oFrame.at<double>(1, i)/bound_oFrame.at<double>(2, i);
		if(xtmp < xmin)
			xmin = xtmp;
		if(xtmp > xmax)
			xmax = xtmp;
		if(ytmp < ymin)
			ymin = ytmp;
		if(ytmp > ymax)
			ymax = ytmp;
	}

	// calculate ratio and scale
	aspect_ratio = (xmax - xmin)/(ymax - ymin);
	height = frame_size.height;
	width = (int)height*aspect_ratio;
	scale = height/(ymax - ymin);

	oFrame = Mat(Size(width, height), iFrame.type());

	// set the new H inverse with offsets
	hinv.at<double>(0,2) = scale*(xmin*hinv.at<double>(0,0) + ymin*hinv.at<double>(0,1) + hinv.at<double>(0,2));
	hinv.at<double>(1,2) = scale*(xmin*hinv.at<double>(1,0) + ymin*hinv.at<double>(1,1) + hinv.at<double>(1,2));
	hinv.at<double>(2,2) = scale*(xmin*hinv.at<double>(2,0) + ymin*hinv.at<double>(2,1) + hinv.at<double>(2,2));

	for(int i = 0; i < width; ++i)
	{
		tmpPoint.x = (double)i;
		for(int j = 0; j < height; ++j)
		{
			tmpPoint.y = (double)j;
			tmpMat = convert2Homogeneous(tmpPoint);
			tmpResult = hinv*tmpMat;
			double resultx = tmpResult.at<double>(0,0)/tmpResult.at<double>(2,0);
			double resulty = tmpResult.at<double>(1,0)/tmpResult.at<double>(2,0);

			if(resultx < 0 || resultx > frame_size.width - 1 || resulty < 0 || resulty > frame_size.height - 1)
				continue;

			// apply bilinear interpollation to smooth R, G, B channels of the target frame
			for(int k = 0; k < 3; ++k)
			{
				double sum = 0;
				sum += (1.0 - (resultx - (int)resultx))*(1.0 - (resulty - (int)resulty))*iFrame.data[iFrame.channels()*(iFrame.cols*(int)resulty + (int)resultx) + k];
				sum += (1.0 - (resultx - (int)resultx))*(resulty - (int)resulty)*iFrame.data[iFrame.channels()*(iFrame.cols*(int)(resulty + 1) + (int)resultx) + k];
				sum += (resultx - (int)resultx)*(1.0 - (resulty - (int)resulty))*iFrame.data[iFrame.channels()*(iFrame.cols*(int)resulty + (int)(resultx + 1)) + k];
				sum += (resultx - (int)resultx)*(resulty - (int)resulty)*iFrame.data[iFrame.channels()*(iFrame.cols*(int)(resulty + 1) + (int)(resultx + 1)) + k];
				oFrame.data[oFrame.channels()*(oFrame.cols*j + i) + k] = sum;
			}
		}
	}

	return hinv;

}

/*------------------------------------------------
 * Function: 
 *		projectCorrectH - correct the projective  
 *		distortion by projecting vanishing line 
 *		to the infinity line
 * Input:
 *		
 * Output:
 *		
 *------------------------------------------------
 */

Mat projectCorrectH(const Mat VL)
{
	Mat Hp = Mat::eye(3, 3, CV_64F);
	Hp.at<double>(2,0) = VL.at<double>(0,0);
	Hp.at<double>(2,1) = VL.at<double>(1,0);
	Hp.at<double>(2,2) = VL.at<double>(2,0);

	return Hp;
}

/*------------------------------------------------
 * Function: 
 *		affineCorrectH - correct the affine  
 *		distortion based on corrected projective 
 *		distortion plane
 * Input:
 *		
 * Output:
 *		
 *------------------------------------------------
 */

Mat affineCorrectH(const Mat Hp, Mat l1, Mat l2, Mat l3, Mat l4)
{
	Mat Hpinv = Hp.inv();
	Mat Hpit = Hpinv.t();
	
	l1 = Hpit*l1;
	l2 = Hpit*l2;
	l3 = Hpit*l3;
	l4 = Hpit*l4;

	double Ldata[2][2] = {{l1.at<double>(0,0)*l2.at<double>(0,0),
		l1.at<double>(0,0)*l2.at<double>(0,0) + l1.at<double>(1,0)*l2.at<double>(0,0)},
		{l3.at<double>(0,0)*l4.at<double>(0,0), l3.at<double>(0,0)*l4.at<double>(0,0) +
		l3.at<double>(1,0)*l4.at<double>(0,0)}};

	double bdata[2][1] = {{-l1.at<double>(1,0)*l2.at<double>(1,0)}, {-l3.at<double>(1,0)*l4.at<double>(1,0)}};

	Mat L(2, 2, CV_64F, Ldata);
	Mat b(2, 1, CV_64F, bdata);

	Mat s = L.inv()*b;

	double Sdata[2][2] = {{s.at<double>(0,0), s.at<double>(1,0)}, {s.at<double>(1,0), 1.0}};

	Mat S = Mat(2, 2, CV_64F, Sdata);

	Mat V, D2, D, Vt;
	SVD::compute(S, D2, V, Vt, 0);

	sqrt(D2, D);
	D = Mat::diag(D);

	Mat A = V*D*Vt;

	Mat Ha = Mat::eye(3, 3, CV_64F);

	Ha.at<double>(0,0) = A.at<double>(0,0);
	Ha.at<double>(0,1) = A.at<double>(0,1);
	Ha.at<double>(1,0) = A.at<double>(1,0);
	Ha.at<double>(1,1) = A.at<double>(1,1);
	
	return Ha;
}


/*------------------------------------------------
 * Function: 
 *		ODrawChosenLine - 1 step method to draw
 *		chosen lines on the original image given
 *		several points
 * Input:
 *		
 * Output:
 *		
 *------------------------------------------------
 */

void ODrawChosenLine(Mat& image, const Mat pt1, const Mat pt2,
	const Mat pt3, const Mat pt4, const Mat pt5, const Mat pt6,
	const Mat pt7, const Mat pt8, const Mat pt9, const Mat pt10,
	const Mat pt11, const Mat pt12, const Mat pt13, const Mat pt14,
	const Mat pt15)
{
	Point2d tpt1(pt1.at<double>(0), pt1.at<double>(1));
	Point2d tpt2(pt2.at<double>(0), pt2.at<double>(1));
	Point2d tpt3(pt3.at<double>(0), pt3.at<double>(1));
	Point2d tpt4(pt4.at<double>(0), pt4.at<double>(1));
	Point2d tpt5(pt5.at<double>(0), pt5.at<double>(1));
	Point2d tpt6(pt6.at<double>(0), pt6.at<double>(1));
	Point2d tpt7(pt7.at<double>(0), pt7.at<double>(1));
	Point2d tpt8(pt8.at<double>(0), pt8.at<double>(1));
	Point2d tpt9(pt9.at<double>(0), pt9.at<double>(1));
	Point2d tpt10(pt10.at<double>(0), pt10.at<double>(1));
	Point2d tpt11(pt11.at<double>(0), pt11.at<double>(1));
	Point2d tpt12(pt12.at<double>(0), pt12.at<double>(1));
	Point2d tpt13(pt13.at<double>(0), pt13.at<double>(1));
	Point2d tpt14(pt14.at<double>(0), pt14.at<double>(1));
	Point2d tpt15(pt15.at<double>(0), pt15.at<double>(1));

	line(image, tpt1, tpt2, Scalar(0, 0, 255), 2);
	line(image, tpt2, tpt3, Scalar(0, 0, 255), 2);
	line(image, tpt4, tpt5, Scalar(255, 0, 0), 2);
	line(image, tpt5, tpt6, Scalar(255, 0, 0), 2);
	line(image, tpt7, tpt8, Scalar(0, 255, 0), 2);
	line(image, tpt8, tpt9, Scalar(0, 255, 0), 2);
	line(image, tpt10, tpt11, Scalar(255, 255, 0), 2);
	line(image, tpt11, tpt12, Scalar(255, 255, 0), 2);
	line(image, tpt13, tpt14, Scalar(255, 0, 255), 2);
	line(image, tpt14, tpt15, Scalar(255, 0, 255), 2);
}


/*---------------------------------------------------
 * Function: 
 *		OneStepH - find the homography transformation  
 *		by one step method
 * Input:
 *		
 * Output:
 *		
 *---------------------------------------------------
 */

Mat OneStepH(Mat l1, Mat l2, Mat l3, Mat l4, Mat l5,
	Mat l6, Mat l7, Mat l8, Mat l9, Mat l10)
{

	double wdata[5][5] = {
	{
		l1.at<double>(0,0)*l2.at<double>(0,0),
		0.5*(l1.at<double>(0,0)*l2.at<double>(1,0)+l1.at<double>(1,0)*l2.at<double>(0,0)),
		l1.at<double>(1,0)*l2.at<double>(1,0), 
		0.5*(l1.at<double>(0,0)*l2.at<double>(2,0)+l1.at<double>(2,0)*l2.at<double>(0,0)),
		0.5*(l1.at<double>(1,0)*l2.at<double>(2,0)+l1.at<double>(2,0)*l2.at<double>(1,0))
	},
	{
		l3.at<double>(0,0)*l4.at<double>(0,0),
		0.5*(l3.at<double>(0,0)*l4.at<double>(1,0)+l3.at<double>(1,0)*l4.at<double>(0,0)),
		l3.at<double>(1,0)*l4.at<double>(1,0), 
		0.5*(l3.at<double>(0,0)*l4.at<double>(2,0)+l3.at<double>(2,0)*l4.at<double>(0,0)),
		0.5*(l3.at<double>(1,0)*l4.at<double>(2,0)+l3.at<double>(2,0)*l4.at<double>(1,0))
	},
	{
		l5.at<double>(0,0)*l6.at<double>(0,0),
		0.5*(l5.at<double>(0,0)*l6.at<double>(1,0)+l5.at<double>(1,0)*l6.at<double>(0,0)),
		l5.at<double>(1,0)*l6.at<double>(1,0), 
		0.5*(l5.at<double>(0,0)*l6.at<double>(2,0)+l5.at<double>(2,0)*l6.at<double>(0,0)),
		0.5*(l5.at<double>(1,0)*l6.at<double>(2,0)+l5.at<double>(2,0)*l6.at<double>(1,0))
	},
	{
		l7.at<double>(0,0)*l8.at<double>(0,0),
		0.5*(l7.at<double>(0,0)*l8.at<double>(1,0)+l7.at<double>(1,0)*l8.at<double>(0,0)),
		l7.at<double>(1,0)*l8.at<double>(1,0), 
		0.5*(l7.at<double>(0,0)*l8.at<double>(2,0)+l7.at<double>(2,0)*l8.at<double>(0,0)),
		0.5*(l7.at<double>(1,0)*l8.at<double>(2,0)+l7.at<double>(2,0)*l8.at<double>(1,0))
	},
	{
		l9.at<double>(0,0)*l10.at<double>(0,0),
		0.5*(l9.at<double>(0,0)*l10.at<double>(1,0)+l9.at<double>(1,0)*l10.at<double>(0,0)),
		l9.at<double>(1,0)*l10.at<double>(1,0), 
		0.5*(l9.at<double>(0,0)*l10.at<double>(2,0)+l9.at<double>(2,0)*l10.at<double>(0,0)),
		0.5*(l9.at<double>(1,0)*l10.at<double>(2,0)+l9.at<double>(2,0)*l10.at<double>(1,0))
	}
	};
	
	double bdata[5][1] = {
		{-l1.at<double>(2,0)*l2.at<double>(2,0)},
		{-l3.at<double>(2,0)*l4.at<double>(2,0)},
		{-l5.at<double>(2,0)*l6.at<double>(2,0)},
		{-l7.at<double>(2,0)*l8.at<double>(2,0)},
		{-l9.at<double>(2,0)*l10.at<double>(2,0)}
	};	

	Mat w(5, 5, CV_64F, wdata);
	Mat b(5, 1, CV_64F, bdata);
	Mat z = w.inv()*b;
	
	double Cinfdata[3][3] = {
		{z.at<double>(0,0), 0.5*z.at<double>(1,0), 0.5*z.at<double>(3,0)},
		{0.5*z.at<double>(1,0), z.at<double>(2,0), 0.5*z.at<double>(4,0)},
		{0.5*z.at<double>(3,0), 0.5*z.at<double>(4,0), 1}
	};
	
	Mat Cinf(3, 3, CV_64F, Cinfdata);

	cout << "Cinf: " << Cinf << endl;

	Mat V, D, D2, Vt;

	Mat AAt = Cinf(Range(0, 2), Range(0, 2));
	SVD::compute(AAt, D2, V, Vt, 0);

	cout << "AAt: " << AAt << endl;

	sqrt(D2, D);
	D = Mat::diag(D);

	cout << "D: " << D << endl;

	Mat A = V*D*Vt;
	cout << "A: " << A << endl;

	double rdata[2][1] = {{Cinf.at<double>(0,2)},{Cinf.at<double>(1,2)}};
	Mat r(2, 1, CV_64F, rdata);

	Mat v;
	solve(A, r, v, DECOMP_SVD);
	
	Mat H = Mat::eye(3, 3, CV_64F);

	H.at<double>(0,0) = A.at<double>(0,0);
	H.at<double>(0,1) = A.at<double>(0,1);
	H.at<double>(1,0) = A.at<double>(1,0);
	H.at<double>(1,1) = A.at<double>(1,1);
	H.at<double>(2,0) = v.at<double>(0,0);
	H.at<double>(2,1) = v.at<double>(1,0);

	return H;
	
}


int main()
{
	//char* imgName = "Set1/Img1.jpg";
	//char* dataName = "set1Img1.yml";
	char* imgName = "Set1/Img2.jpg";
	char* dataName = "set1Img2.yml";

	Mat img = imread(imgName);

	Size frame_size = img.size();

	Mat x1, x2, x3, x4, x5, x6, x7, x8;
	FileStorage fs(dataName, CV_STORAGE_READ);
	if(!fs.isOpened())
	{
		cout << "Failed to open data file" << endl;
		return 0;
	}
	fs["x1"] >> x1;
	fs["x2"] >> x2;
	fs["x3"] >> x3;
	fs["x4"] >> x4;
	fs["x5"] >> x5;
	fs["x6"] >> x6;
	fs["x7"] >> x7;
	fs["x8"] >> x8;

	TDrawChosenLine(img, x1, x2, x3, x4, x5, x6, x7, x8);
	
	Mat l1 = constructLine(x1, x2);
	Mat l2 = constructLine(x3, x4);
	Mat l3 = constructLine(x1, x3);
	Mat l4 = constructLine(x2, x4);
	Mat l5 = constructLine(x5, x6);
	Mat l6 = constructLine(x7, x8);

	Mat P = constructPoint(l1, l2);
	Mat Q = constructPoint(l3, l4);

	Mat VL = constructLine(P, Q);

	Mat Hp = projectCorrectH(VL);

	cout << "Hp: " << Hp << endl;
	
	Mat imgProjCorrected, imgAffineCorrected;

	imageBackProj(img, imgProjCorrected, frame_size, Hp);

	Mat Ha = affineCorrectH(Hp, l1, l3, l5, l6);

	cout << "Ha: " << Ha << endl;

	imageBackProj(img, imgAffineCorrected, frame_size, Ha.inv()*Hp); 

	namedWindow("TwoStep_Img", WINDOW_AUTOSIZE);
	namedWindow("TwoStep_Img_final", WINDOW_AUTOSIZE);
	imshow("TwoStep_Img", imgProjCorrected);
	imshow("TwoStep_Img_final", imgAffineCorrected);
	imwrite("result1.jpg", imgProjCorrected);
	imwrite("result2.jpg", imgAffineCorrected);
	

	/*----------------------------
	 * One Step Method
	 *----------------------------
	 */

	
	img = imread(imgName);

	Mat ox1, ox2, ox3, ox4, ox5, ox6, ox7, ox8, ox9, ox10, ox11, ox12, ox13, ox14, ox15;

	fs["ox1"] >> ox1;
	fs["ox2"] >> ox2;
	fs["ox3"] >> ox3;
	fs["ox4"] >> ox4;
	fs["ox5"] >> ox5;
	fs["ox6"] >> ox6;
	fs["ox7"] >> ox7;
	fs["ox8"] >> ox8;
	fs["ox9"] >> ox9;
	fs["ox10"] >> ox10;
	fs["ox11"] >> ox11;
	fs["ox12"] >> ox12;
	fs["ox13"] >> ox13;
	fs["ox14"] >> ox14;
	fs["ox15"] >> ox15;

	ODrawChosenLine(img, ox1, ox2, ox3, ox4, ox5, ox6, ox7, ox8, ox9, ox10, ox11, ox12, ox13, ox14, ox15);

	Mat ol1 = constructLine(ox1, ox2);
	Mat ol2 = constructLine(ox2, ox3);
	Mat ol3 = constructLine(ox4, ox5);
	Mat ol4 = constructLine(ox5, ox6);
	Mat ol5 = constructLine(ox7, ox8);
	Mat ol6 = constructLine(ox8, ox9);
	Mat ol7 = constructLine(ox10, ox11);
	Mat ol8 = constructLine(ox11, ox12);
	Mat ol9 = constructLine(ox13, ox14);
	Mat ol10 = constructLine(ox14, ox15);
	
	Mat H = OneStepH(ol1, ol2, ol3, ol4, ol5, 
		ol6, ol7, ol8, ol9, ol10);

	cout << "H: " << H << endl;

	Mat imgOneCorrected;
	imageBackProj(img, imgOneCorrected, frame_size, H.inv());

	namedWindow("OneStep_Img", WINDOW_AUTOSIZE);
	imshow("OneStep_Img", imgOneCorrected);
	imwrite("result3.jpg", imgOneCorrected);

	waitKey(0);
	fs.release();

	return 0;
}