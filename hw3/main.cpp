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

void TDrawChosenLine(Mat& image, const Point2d pt1, const Point2d pt2,
	const Point2d pt3,const Point2d pt4, const Point2d pt5,
	const Point2d pt6,const Point2d pt7, const Point2d pt8)
{
	line(image, pt1, pt2, Scalar(0, 0, 255), 2);
	line(image, pt3, pt4, Scalar(0, 0, 255), 2);
	line(image, pt1, pt3, Scalar(255, 0, 0), 2);
	line(image, pt2, pt4, Scalar(255, 0, 0), 2);
	line(image, pt5, pt6, Scalar(0, 255, 0), 2);
	line(image, pt7, pt8, Scalar(0, 255, 0), 2);
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

void ODrawChosenLine(Mat& image, const Point2d pt1, const Point2d pt2,
	const Point2d pt3, const Point2d pt4, const Point2d pt5, const Point2d pt6,
	const Point2d pt7, const Point2d pt8, const Point2d pt9, const Point2d pt10,
	const Point2d pt11, const Point2d pt12, const Point2d pt13, const Point2d pt14,
	const Point2d pt15, const Point2d pt16)
{
	line(image, pt1, pt2, Scalar(0, 0, 255), 2);
	line(image, pt3, pt4, Scalar(0, 0, 255), 2);
	line(image, pt5, pt6, Scalar(255, 0, 0), 2);
	line(image, pt6, pt7, Scalar(255, 0, 0), 2);
	line(image, pt8, pt9, Scalar(0, 255, 0), 2);
	line(image, pt9, pt10, Scalar(0, 255, 0), 2);
	line(image, pt11, pt12, Scalar(255, 255, 0), 2);
	line(image, pt12, pt13, Scalar(255, 255, 0), 2);
	line(image, pt14, pt15, Scalar(255, 0, 255), 2);
	line(image, pt15, pt16, Scalar(255, 0, 255), 2);
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
	}};

	double bdata[5][1] = {
		{-l1.at<double>(2,0)*l2.at<double>(2,0)},
		{-l3.at<double>(2,0)*l4.at<double>(2,0)},
		{-l5.at<double>(2,0)*l6.at<double>(2,0)},
		{-l7.at<double>(2,0)*l8.at<double>(2,0)},
		{-l9.at<double>(2,0)*l10.at<double>(2,0)}
	};

	Mat w(5, 5, CV_64F, wdata);
	Mat b(5, 1, CV_64F, bdata);

	cout << "w: " << w.inv() << endl;

	Mat z = w.inv(DECOMP_SVD)*b;

	cout << "l9: " << l1 << endl;
	cout << "l10: " << l2 << endl;

	cout << "z: " << z << endl;

	double Cinfdata[3][3] = {
		{z.at<double>(0,0), 0.5*z.at<double>(1,0), 0.5*z.at<double>(3,0)},
		{0.5*z.at<double>(1,0), z.at<double>(2,0), 0.5*z.at<double>(4,0)},
		{0.5*z.at<double>(3,0), 0.5*z.at<double>(4,0), 1.0}
	};

	Mat Cinf(3, 3, CV_64F, Cinfdata);

	cout << "Cinf: " << Cinf << endl;

	Mat V, D, D2, Vt;
	SVD::compute(Cinf, D, V, Vt, 0);

	cout << "D: " << D << endl;

	Mat AAt = Cinf(Range(0, 2), Range(0, 2));
	SVD::compute(AAt, D2, V, Vt, 0);

	cout << "AAt: " << AAt << endl;

	sqrt(D2, D);
	D = Mat::diag(D);

	//cout << "D: " << D << endl;

	Mat A = V*D*Vt;
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

	char* set1Img1Name = "Set1/Img1.jpg";
	char* set1Img2Name = "Set1/Img2.jpg";

	Mat set1Img1 = imread(set1Img1Name);
	//Mat set1Img2 = imread(set1Img2Name)

	Point2d pt1(599.0, 201.0);
	Point2d pt2(669.0, 225.0);
	Point2d pt3(565.0, 446.0);
	Point2d pt4(641.0, 438.0);
	Point2d pt5(603.0, 212.0);
	Point2d pt6(643.0, 390.0);
	Point2d pt7(664.0, 233.0);
	Point2d pt8(579.0, 389.0);

	TDrawChosenLine(set1Img1, pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8);

	Mat x1 = convert2Homogeneous(pt1);
	Mat x2 = convert2Homogeneous(pt2);
	Mat x3 = convert2Homogeneous(pt3);
	Mat x4 = convert2Homogeneous(pt4);
	Mat x5 = convert2Homogeneous(pt5);
	Mat x6 = convert2Homogeneous(pt6);
	Mat x7 = convert2Homogeneous(pt7);
	Mat x8 = convert2Homogeneous(pt8);

	
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

	Size frame_size = set1Img1.size();
	
	Mat set1Img1_corrected, set1Img1_corrected2;

	imageBackProj(set1Img1, set1Img1_corrected, frame_size, Hp);

	Mat Ha = affineCorrectH(Hp, l1, l3, l5, l6);

	cout << "l1: " << l1 << endl;
	cout << "l5: " << l5 << endl;

	imageBackProj(set1Img1, set1Img1_corrected2, frame_size, Ha.inv()*Hp); 

	namedWindow("TwoStep_set1Img1", WINDOW_AUTOSIZE);
	namedWindow("TwoStep_set1Img1_final", WINDOW_AUTOSIZE);
	imshow("TwoStep_set1Img1", set1Img1_corrected);
	imshow("TwoStep_set1Img1_final", set1Img1_corrected2);


	/*----------------------------
	 * First Step Method
	 *----------------------------
	 */


	set1Img1 = imread(set1Img1Name);

	//5 pairs of lines data 

	Point2d opt1(603.0, 212.0);
	Point2d opt2(643.0, 390.0);
	Point2d opt3(664.0, 233.0);
	Point2d opt4(579.0, 389.0);

	Point2d opt5(599.0, 201.0);
	Point2d opt6(565.0, 446.0);
	Point2d opt7(641.0, 438.0);

	Point2d opt8(283.0, 12.0);
	Point2d opt9(511.0, 113.0);
	Point2d opt10(473.0, 394.0);

	Point2d opt11(322.0, 93.0);
	//Point2d opt12(271.0, 454.0);
	Point2d opt12(470.0, 148.0);
	Point2d opt13(431.0, 440.0);

	Point2d opt14(696.0, 235.0);
	Point2d opt15(670.0, 434.0);
	Point2d opt16(722.0, 429.0);

	//Draw Chosen five pairs of lines
	ODrawChosenLine(set1Img1, opt1, opt2, opt3, opt4, opt5, opt6, opt7, opt8, opt9, opt10, opt11, opt12, opt13, opt14, opt15, opt16);

	Mat ox1 = convert2Homogeneous(opt1);
	Mat ox2 = convert2Homogeneous(opt2);
	Mat ox3 = convert2Homogeneous(opt3);
	Mat ox4 = convert2Homogeneous(opt4);
	Mat ox5 = convert2Homogeneous(opt5);
	Mat ox6 = convert2Homogeneous(opt6);
	Mat ox7 = convert2Homogeneous(opt7);
	Mat ox8 = convert2Homogeneous(opt8);
	Mat ox9 = convert2Homogeneous(opt9);
	Mat ox10 = convert2Homogeneous(opt10);
	Mat ox11 = convert2Homogeneous(opt11);
	Mat ox12 = convert2Homogeneous(opt12);
	Mat ox13 = convert2Homogeneous(opt13);
	Mat ox14 = convert2Homogeneous(opt14);
	Mat ox15 = convert2Homogeneous(opt15);
	Mat ox16 = convert2Homogeneous(opt16);

	Mat ol1 = constructLine(ox1, ox2);
	Mat ol2 = constructLine(ox3, ox4);
	Mat ol3 = constructLine(ox5, ox6);
	Mat ol4 = constructLine(ox6, ox7);
	Mat ol5 = constructLine(ox8, ox9);
	Mat ol6 = constructLine(ox9, ox10);
	Mat ol7 = constructLine(ox11, ox12);
	Mat ol8 = constructLine(ox12, ox13);
	Mat ol9 = constructLine(ox14, ox15);
	Mat ol10 = constructLine(ox15, ox16);
	
	Mat H = OneStepH(ol1, ol2, ol3, ol4, ol5, 
		ol6, ol7, ol8, ol9, ol10);

	Mat set1Img1_corrected3;
	imageBackProj(set1Img1, set1Img1_corrected3, frame_size, H.inv());

	cout << "H inverse 1: " << Ha.inv()*Hp << endl;
	cout << "H inverse 2: " << H.inv() << endl;

	namedWindow("OneStep_set1Img1", WINDOW_AUTOSIZE);
	imshow("OneStep_set1Img1", set1Img1_corrected3);

	waitKey(0);

	return 0;
}