#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <opencv.hpp>
#include <levmar.h>

using namespace std;
using namespace cv;

typedef struct{
	Point2d *pt1;
	Point2d *pt2;
	Point3d *X;
	int num;
}PTSPAIRS;

#define MAX_INTEREST_PT 500
#define WSIZE 15

/*------------------------------------------------
 * Function: 
 *		drawPairs - draw corresponding line pairs
 *		for two
 * Input:
 *		img1: the first image with points on it
 *		img2: the second image with points on it
 *		imgOut: the final merged image with 
 *		matching lines on it
 *		pt1: container including points in the
 *		left image
 *		pt2: container including points in the 
 *		right image
 * Output:
 *		Mat imgOut
 *------------------------------------------------
 */

void drawPairs(const Mat& img1, const Mat& img2, Mat& imgOut, const vector<Point2d>& pt1, const vector<Point2d>& pt2)
{
	// matching line color space
	Scalar Color[7] = {Scalar(255, 0, 255), Scalar(255, 255, 255), Scalar(0, 255, 0), Scalar(0, 255, 255),
	Scalar(255, 0, 0), Scalar(255, 255, 0), Scalar(0, 0, 255)};
	Size size = img1.size();
	// put two images to one
	img1.copyTo(imgOut(Rect(0, 0, size.width, size.height)));
	img2.copyTo(imgOut(Rect(size.width, 0, size.width, size.height)));
	// draw lines
	for(size_t i = 0; i < pt1.size(); i++) 
	{
		line(imgOut, pt1[i], Point(pt2[i].x + size.width, pt2[i].y), Color[i%7]);
	}
}


/*------------------------------------------------
 * Function: 
 *		normalizeX - normalize points and return 
 *		normalization matrix T
 * Input:
 *		numpairs: number of point pairs 
 *		p: points
 *		T: normalization matrix
 * Output:
 *		T
 *------------------------------------------------
 */

void normalizeX(const int numpairs, Point2d *p, Mat& T)
{
	double scale, tx, ty;
	double meanx, meany;
	double value;
	int i;
	Mat x(3, 1, CV_64FC1);
	Mat xp(3, 1, CV_64FC1);

	meanx = 0;
	meany = 0;
	for(i = 0; i < numpairs; i++)
	{
		meanx += p[i].x;
		meany += p[i].y;
	}
	meanx /= (double)numpairs;
	meany /= (double)numpairs;

	value = 0;
	for(i = 0; i < numpairs; i++)
		value += sqrt(pow(p[i].x - meanx, 2.0) + pow(p[i].y - meany, 2.0));
	value /= (double)numpairs;

	scale = sqrt(2.0)/value;
	tx = -scale*meanx;
	ty = -scale*meany;

	T = Mat::zeros(3, 3, CV_64FC1);
	T.at<double>(0, 0) = scale;
	T.at<double>(0, 2) = tx;
	T.at<double>(1, 1) = scale;
	T.at<double>(1, 2) = ty;
	T.at<double>(2, 2) = 1.0;

	for(i = 0; i < numpairs; i++)
	{
		x.at<double>(0, 0) = p[i].x;
		x.at<double>(1, 0) = p[i].y;
		x.at<double>(2, 0) = 1;
		xp = T*x;
		p[i].x = xp.at<double>(0, 0)/xp.at<double>(2, 0);
		p[i].y = xp.at<double>(1, 0)/xp.at<double>(2, 0);
	}
}


/*------------------------------------------------
 * Function: 
 *		getFundMat - get fundamental matrix 
 * Input:
 *		numpairs: number of point pairs 
 *		pt1: container including points in the
 *		left image
 *		pt2: container including points in the
 *		right image
 * Output:
 *		F
 *------------------------------------------------
 */

Mat getFundMat(const int numpairs, const vector<Point2d>& pt1, const vector<Point2d>& pt2)
{
	int i, j;
	Mat A(numpairs, 9, CV_64FC1, Scalar(1));
	Mat D, U, Vt, V, h, Ftmp, T1, T2, F;
	Mat DD(3, 3, CV_64FC1, Scalar(0)), UU, VVt, VV;
	Point2d *p1 = new Point2d[numpairs];
	Point2d *p2 = new Point2d[numpairs];

	for(i = 0; i < numpairs; i++)
	{
		p1[i].x = pt1[i].x;
		p1[i].y = pt1[i].y;
		p2[i].x = pt2[i].x;
		p2[i].y = pt2[i].y;
	}

	// normalize points
	normalizeX(numpairs, p1, T1);
	normalizeX(numpairs, p2, T2);

	for(i = 0; i < numpairs; i++)
	{
		A.at<double>(i, 0) = p2[i].x*p1[i].x;
		A.at<double>(i, 1) = p2[i].x*p1[i].y;
		A.at<double>(i, 2) = p2[i].x;
		A.at<double>(i, 3) = p2[i].y*p1[i].x;
		A.at<double>(i, 4) = p2[i].y*p1[i].y;
		A.at<double>(i, 5) = p2[i].y;
		A.at<double>(i, 6) = p1[i].x;
		A.at<double>(i, 7) = p1[i].y;
		A.at<double>(i, 8) = 1;
	}

	SVD::compute(A, D, U, Vt, 0);
	V = Vt.t();
	h = V.col(8);
	// normalize the smallest eigen vector
	for(i = 0; i < 8; i++)
		h.at<double>(i, 0) = h.at<double>(i, 0)/h.at<double>(8, 0);
	h.at<double>(8, 0) = 1;

	// normalized F
	Ftmp = Mat(h.t()).reshape(0, 3);
	SVD::compute(Ftmp, D, UU, VVt, 0);
	// set the smallest eigen value to 0
	DD.at<double>(0, 0) = D.at<double>(0);
	DD.at<double>(1, 1) = D.at<double>(1);
	Ftmp = UU*DD;
	Ftmp = Ftmp*VVt;
	// get true F for original points
	F = T2.t()*Ftmp;
	F = F*T1;
	for(i = 0; i < 3; i++)
		for(j = 0; j < 3; j++)
			F.at<double>(i, j) = F.at<double>(i, j)/F.at<double>(2, 2);

	cout << "F: " << F << endl;
	delete p1;
	delete p2;

	return F; 
}


/*------------------------------------------------
 * Function: 
 *		computeEpp - compute epipoles
 * Input:
 *		F: calculated fundamental matrix 
 *		e: epipole for the left image
 *		ep: epipole for the right image
 * Output:
 *		e, ep
 *------------------------------------------------
 */

void computeEpp(const Mat F, Mat& e, Mat& ep)
{
	Mat D, Ut, U, V, Vt;

	e = Mat::zeros(3, 1, CV_64FC1);
	ep = Mat::zeros(3, 1, CV_64FC1);
	SVD::compute(F, D, U, Vt, 0);
	Ut = U.t();
	V = Vt.t();
	// right null vector
	e = V.col(2);
	// left null vector
	ep = U.col(2);
	for(int i = 0; i < 3; i++)
	{
		e.at<double>(i, 0) = e.at<double>(i, 0)/e.at<double>(2, 0);
		ep.at<double>(i, 0) = ep.at<double>(i, 0)/ep.at<double>(2, 0);
	}
	cout << "e: " << e << endl;
	cout << "ep: " << ep << endl;
}


/*------------------------------------------------
 * Function: 
 *		vec2skew - convert a vector to skew matrix
 * Input:
 *		v: input vector 
 *		vx: output skew matrix
 * Output:
 *		vx
 *------------------------------------------------
 */

void vec2skew(const Mat& v, Mat& vx)
{
	vx = Mat::zeros(3, 3, CV_64FC1);
	vx.at<double>(0, 1) = -v.at<double>(2, 0);
	vx.at<double>(0, 2) = v.at<double>(1, 0);
	vx.at<double>(1, 0) = v.at<double>(2, 0);
	vx.at<double>(1, 2) = -v.at<double>(0, 0);
	vx.at<double>(2, 0) = -v.at<double>(1, 0);
	vx.at<double>(2, 1) = v.at<double>(0, 0);
}


/*------------------------------------------------
 * Function: 
 *		computeP - compute camera matrix for two
 *		views, canonical configuration
 * Input:
 *		F: fundamental matrix 
 *		ep: epipole for the right view
 *		P: camera matrix for the left view
 *		Pp: camera matrix for the right view
 * Output:
 *		P, Pp
 *------------------------------------------------
 */

void computeP(const Mat& F, const Mat& ep, Mat& P, Mat& Pp)
{
	Mat epx, tmp;
	int i = 0, j = 0;

	P = Mat::zeros(3, 4, CV_64FC1);
	Pp = Mat::zeros(3, 4, CV_64FC1);

	// canonical configuration
	P.at<double>(0, 0) = 1;
	P.at<double>(1, 1) = 1;
	P.at<double>(2, 2) = 1;

	vec2skew(ep, epx);
	// F multiply any skew matrix
	tmp = epx*F;
	for(i = 0; i < 3; i++)
		for(j = 0; j < 3; j++)
			Pp.at<double>(i, j) = tmp.at<double>(i, j);
	// the last column is epipole for the right view
	for(i = 0; i < 3; i++)
		Pp.at<double>(i, 3) = ep.at<double>(i, 0);

	cout << "P: " << P << endl;
	cout << "Pp: " << Pp << endl;
}


/*------------------------------------------------
 * Function: 
 *		computeEpLines - compute epipolar lines
 * Input:
 *		imgSize: size of the image 
 *		F: fundamental matrix
 *		p; the point on the epipolar line
 *		isRight: bit for choosing left or right
 *		epipolar lines
 *		pt1: output point of the intersection with
 *		the boundary
 *		pt2: output point of the intersection with
 *		the boundary
 * Output:
 *		pt1, pt2
 *------------------------------------------------
 */

/*
void computeEpLines(Size imgSize, const Mat& F, Point2d p, int isRight, Point2d& pt1, Point2d& pt2)
{
	int i;
	Point2d inter_bd[4];
	int mask[4];
	Mat x(3, 1, CV_64FC1), l, Ft;
	x.at<double>(0, 0) = p.x;
	x.at<double>(1, 0) = p.y;
	x.at<double>(2, 0) = 1;

	// calculate epipolar lines
	if(isRight){
		l = F*x;
	}else{
		Ft = F.t();
		l = Ft*x;
	}

	inter_bd[0].x = -l.at<double>(2, 0)/l.at<double>(0, 0);
	inter_bd[0].y = 0;
	inter_bd[1].x = -(l.at<double>(2, 0) + l.at<double>(1, 0)*imgSize.height)/l.at<double>(0, 0);
	inter_bd[1].y = imgSize.height;
	inter_bd[2].x = 0;
	inter_bd[2].y = -l.at<double>(2, 0)/l.at<double>(1, 0);
	inter_bd[3].x = imgSize.width;
	inter_bd[3].y = -(l.at<double>(2, 0) + l.at<double>(0, 0)*imgSize.width)/l.at<double>(1, 0);

	for(i = 0; i < 2; i++)
		mask[i] = (inter_bd[i].x >= 0 && inter_bd[i].x < imgSize.width) ? 1 : 0;
	for(i = 2; i < 4; i++)
		mask[i] = (inter_bd[i].y >= 0 && inter_bd[0].y < imgSize.height) ? 1 : 0;

	// determine the intersection of the epipolar line and the boundary 
	i = 0;
	while(mask[i] == 0)
		i++;
	pt1.x = inter_bd[i].x;
	pt1.y = inter_bd[i].y;

	i++;
	while(mask[i] == 0)
		i++;
	pt2.x = inter_bd[i].x;
	pt2.y = inter_bd[i].y;

}
*/

/*------------------------------------------------
 * Function: 
 *		drawEpLines - draw the epipolar lines
 * Input:
 *		pt1: boundary points set 1
 *		pt2: boundary points set 2
 *		imgSize: size of the image
 *		F: fundamental matrix
 *		numpairs: number of point pairs
 *		img1: left image
 *		img2: right image
 * Output:
 *		saved image
 *------------------------------------------------
 */
/*
void drawEpLines(vector<Point2d> pt1, vector<Point2d> pt2, Size imgSize, const Mat& F, int numpairs, const Mat& img1, const Mat& img2)
{
	Mat cimg1 = img1.clone();
	Mat cimg2 = img2.clone();
	Point2d bpt1, bpt2;
	for(int i = 0; i < numpairs; i++)
	{
		circle(cimg1, Point(pt1[i].x, pt1[i].y), 2, Scalar(0, 255, 255), 2, 8, 0);
		computeEpLines(imgSize, F, pt2[i], 0, bpt1, bpt2);
		line(cimg1, Point(bpt1.x, bpt1.y), Point(bpt2.x, bpt2.y), Scalar(0, 0, 255), 1, 8, 0);
		circle(cimg2, Point(pt2[i].x, pt2[i].y), 2, Scalar(0, 255, 255), 2, 8, 0);
		computeEpLines(imgSize, F, pt1[i], 1, bpt1, bpt2);
		line(cimg2, Point(bpt1.x, bpt1.y), Point(bpt2.x, bpt2.y), Scalar(0, 0, 255), 1, 8, 0);
	}
	imwrite("img1eline.jpg", cimg1);
	imwrite("img2eline.jpg", cimg2);
}
*/


/*-----------------------------------------------------
 * Function: 
 *		reconstruct3D - triangulation to reconstruct 
 *		the 3D point
 * Input:
 *		num: number of point pairs 
 *		pt1: points in the left image
 *		pt2: points in the right image
 *		P: camera matrix for the left image
 *		Pp: camera matrix for the right image
 *		X: 3d points container
 * Output:
 *		X
 *-----------------------------------------------------
 */

void reconstruct3D(const int num, const vector<Point2d> pt1, const vector<Point2d> pt2, const Mat P, const Mat Pp, vector<Point3d>& X)
{
	int i, j;
	Mat A(4, 4, CV_64FC1), D, U, V, Vt, h;
	Point3d Xtmp;

	for(i = 0; i < num; i++)
	{
		for(j = 0; j < 4; j++)
		{
			A.at<double>(0, j) = pt1[i].x*P.at<double>(2, j) - P.at<double>(0, j);
			A.at<double>(1, j) = pt1[i].y*P.at<double>(2, j) - P.at<double>(1, j);
			A.at<double>(2, j) = pt2[i].x*Pp.at<double>(2, j) - Pp.at<double>(0, j);
			A.at<double>(3, j) = pt2[i].y*Pp.at<double>(2, j) - Pp.at<double>(1, j);
		}
		SVD::compute(A, D, U, Vt, 0);
		V = Vt.t();
		h = V.col(3);
		Xtmp.x = h.at<double>(0, 0)/h.at<double>(3, 0);
		Xtmp.y = h.at<double>(1, 0)/h.at<double>(3, 0);
		Xtmp.z = h.at<double>(2, 0)/h.at<double>(3, 0);
		X.push_back(Xtmp);
	}
}


/*------------------------------------------------
 * Function: 
 *		errFunc - error function for the LM
 *		optimization
 * Input:
 *		index: index of the point 
 *		x: parameters for optimization
 *		trans_x: parameters for transformed points
 * Output:
 *		trans_x
 *------------------------------------------------
 */

void errFunc(int index, double x[], double trans_x[])
{
	int i, j;
	Mat P(3, 4, CV_64FC1, Scalar(0));
	Mat Pp(3, 4, CV_64FC1, Scalar(0));
	Mat X(4, 1, CV_64FC1, Scalar(0));
	Mat xtmp, xptmp;
	
	// set camera matrices
	P.at<double>(0, 0) = 1;
	P.at<double>(1, 1) = 1;
	P.at<double>(2, 2) = 1;

	for(i = 0; i < 3; i++)
		for(j = 0; j < 4; j++)
			Pp.at<double>(i, j) = x[i*4 + j];

	// get 3d points
	for(j = 0; j < 3; j++)
		X.at<double>(j, 0) = x[12 + index*3 + j];
	X.at<double>(3, 0) = 1;

	// backproject 3d points to image points
	xtmp = P*X;
	trans_x[0] = xtmp.at<double>(0, 0)/xtmp.at<double>(2, 0);
	trans_x[1] = xtmp.at<double>(1, 0)/xtmp.at<double>(2, 0);
	xptmp = Pp*X;
	trans_x[2] = xptmp.at<double>(0, 0)/xptmp.at<double>(2, 0);
	trans_x[3] = xptmp.at<double>(1, 0)/xptmp.at<double>(2, 0);
}


/*-------------------------------------------------
 * Function: 
 *		calcErrFunc - calculate the error function
 *		for all the point pairs
 * Input:
 *		p: parameters for camera matrix for the
 *		right view and 3d points
 *		trans_x: transformed points for image
 *		points
 *		m: parameter number for p
 *		n: parameter number for trans_x
 *		adata: any type argument data
 * Output:
 *		p, trans_x, adata
 *-------------------------------------------------
 */

static void calcErrFunc(double *p, double *trans_x, int m, int n, void *adata)
{
	int i;
	PTSPAIRS *pair;
	pair = (PTSPAIRS *)adata;
	for(i = 0; i < pair->num; i++)
		errFunc(i, p, trans_x + i*4);
}


/*------------------------------------------------
 * Function: 
 *		calcInitErr - calculate the initial error
 *		before optimization
 * Input:
 *		p: parameters for camera matrix for the
 *		right view and 3d points
 *		trans_x: transformed points for image
 *		points
 *		m: parameter number for p
 *		n: parameter number for trans_x
 *		adata: any type argument data
 * Output:
 *		output error
 *------------------------------------------------
 */

void calcInitErr(double *p, double *trans_x, int m, int n, void *adata)
{
	double *tx = new double[n];

	int i;
	PTSPAIRS *pair;
	pair = (PTSPAIRS *)adata;
	for(i = 0; i < 4*pair->num; i++)
		tx[i] = trans_x[i];
	for(i = 0; i < pair->num; i++)
		errFunc(i, p, tx + i*4);
	double error1 = 0, error2 = 0;
	for(i = 0; i < pair->num; i++)
	{
		error1 += pow(tx[4*i] - pair->pt1[i].x, 2.0);
		error1 += pow(tx[4*i + 1] - pair->pt1[i].y, 2.0);
		error2 += pow(tx[4*i + 2] - pair->pt2[i].x, 2.0);
		error2 += pow(tx[4*i + 3] - pair->pt2[i].y, 2.0);
	}
	delete tx;
	cout << "initial error1: " << error1 << endl;
	cout << "initial error2: " << error2 << endl;
}


/*------------------------------------------------
 * Function: 
 *		LMOptPara - LM non-linear optimization for
 *		parameters
 * Input:
 *		numpairs: number of point pairs
 *		p1: points in the left image
 *		p2: points in the right image
 *		Pp: camera matrix for the right view
 *		X: 3d points
 *		e: epipole for the left image
 *		ep: epipole for the right image
 *		F: fundamental matrix
 * Output:
 *		Pp, e, ep, F
 *------------------------------------------------
 */

void LMOptPara(int numpairs, Point2d *p1, Point2d *p2, Mat& Pp, Point3d *X, Mat& e, Mat& ep, Mat& F)
{
	int i, j;
	Mat epx(3, 3, CV_64FC1, Scalar(0));
	Mat M(3, 3, CV_64FC1, Scalar(0)), D, U, V, Vt;
	int ret;
	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0] = LM_INIT_MU; opts[1] = 1E-12; opts[2] = 1E-12; opts[3] = 1E-15;
	opts[4] = LM_DIFF_DELTA;

	// set error function and parameter values
	void (*err)(double *p, double *hx, int m, int n, void* adata);
	int LM_m = 12 + 3*numpairs, LM_n = 4*numpairs;
	double *x = new double[LM_n];
	double *p = new double[LM_m];

	// set structure data argument
	PTSPAIRS ptspairs;
	ptspairs.num = numpairs;
	ptspairs.pt1 = p1;
	ptspairs.pt2 = p2;
	ptspairs.X = X;

	// set parameters
	for(i = 0; i < 3; i++)
		for(j = 0; j < 4; j++)
			p[4*i + j] = Pp.at<double>(i, j);

	for(i = 0; i < numpairs; i++)
	{
		p[12 + i*3] = X[i].x;
		p[12 + i*3 + 1] = X[i].y;
		p[12 + i*3 + 2] = X[i].z;
	}

	for(i = 0; i < numpairs; i++)
	{
		x[4*i] = ptspairs.pt1[i].x;
		x[4*i + 1] = ptspairs.pt1[i].y;
		x[4*i + 2] = ptspairs.pt2[i].x;
		x[4*i + 3] = ptspairs.pt2[i].y;
	}

	// show the initial error
	calcInitErr(p, x, LM_m, LM_n, &ptspairs);

	err = calcErrFunc;
	// run optimization
	ret = dlevmar_dif(err, p, x, LM_m, LM_n, 1000, opts, info, NULL, NULL, &ptspairs);
	printf("error for x and y: %f %f\n", info[0], info[1]);
	printf("LM algorithm iterations: %f \n", info[5]);

	// update matrices
	for(i = 0; i < 3; i++)
		for(j = 0; j < 4; j++)
			Pp.at<double>(i, j) = p[4*i + j];

	for(i = 0; i < numpairs; i++)
	{
		X[i].x = p[12 + i*3];
		X[i].y = p[12 + i*3 + 1];
		X[i].z = p[12 + i*3 + 2];
	}

	for(i = 0; i < 3; i++)
		ep.at<double>(i, 0) = Pp.at<double>(i, 3);

	vec2skew(ep, epx);
	for(i = 0; i < 3; i++)
		for(j = 0; j < 3; j++)
			M.at<double>(i, j) = Pp.at<double>(i, j);
	F = epx*M;
	for(i = 0; i < 3; i++)
		for(j = 0; j < 3; j++)
			F.at<double>(i, j) = F.at<double>(i, j)/F.at<double>(2, 2);

	cout << "after opt F: " << F << endl;

	SVD::compute(F, D, U, Vt, 0);
	V = Vt.t();
	e = V.col(2);

	delete x;
	delete p;
}


/*----------------------------------------------------
 * Function: 
 *		ptsTransform - transform homogeneous point by 
 *		homography
 * Input:
 *		H: homography matrix 
 *		x: original point
 *		xnew: new point
 * Output:
 *		xnew
 *----------------------------------------------------
 */

void ptsTransform(const Mat H, Point2d& x, Point2d& xnew)
{
	Mat ptx(3, 1, CV_64FC1, Scalar(1));
	Mat ptxnew(3, 1, CV_64FC1, Scalar(1));

	ptx.at<double>(0, 0) = x.x;
	ptx.at<double>(1, 0) = x.y;
	ptxnew = H*ptx;
	xnew.x = ptxnew.at<double>(0, 0)/ptxnew.at<double>(2, 0);
	xnew.y = ptxnew.at<double>(1, 0)/ptxnew.at<double>(2, 0);
}


/*------------------------------------------------
 * Function: 
 *		rectifyImg - rectify the image
 * Input:
 *		imgSize: size of the image 
 *		num: number of point pairs:
 *		pt1: points in the left image
 *		pt2: points in the right image
 *		ep: epipole for the right view
 *		P: camera matrix for the left view
 *		Pp: camera matrix for the right view
 *		H: homography matrix for the left image
 *		Hp: homography matrix for the right image
 *		F:
 * Output:
 *		H, Hp
 *------------------------------------------------
 */

/*
void rectifyImg(Size imgSize, int num, vector<Point2d> pt1, vector<Point2d> pt2, Mat ep, Mat P, Mat Pp, Mat& H, Mat& Hp, Mat F)
{
	int i;
	int centerx, centery;
	double alpha, f;
	Mat T(3, 3, CV_64FC1, Scalar(0)), T2(3, 3, CV_64FC1, Scalar(0)), Ha(3, 3, CV_64FC1, Scalar(0));
	Mat R(3, 3, CV_64FC1, Scalar(0));
	Mat G(3, 3, CV_64FC1, Scalar(0));
	Mat A(num, 3, CV_64FC1, Scalar(0)), b(num, 1, CV_64FC1, Scalar(0));
	Mat Pinv, M, H0, t;
	Point2d xnew, xpnew;

	centerx = imgSize.width/2;
	centery = imgSize.height/2;

	T.at<double>(0, 0) = 1;
	T.at<double>(1, 1) = 1;
	T.at<double>(2, 2) = 1;
	T.at<double>(0, 2) = -centerx;
	T.at<double>(1, 2) = -centery;

	alpha = atan(-(ep.at<double>(1, 0)/ep.at<double>(2, 0) - centery)/(ep.at<double>(0, 0)/ep.at<double>(2, 0) - centerx));
	f = cos(alpha)*(ep.at<double>(0, 0)/ep.at<double>(2, 0) - centerx) - sin(alpha)*(ep.at<double>(1, 0)/ep.at<double>(2, 0) - centery);
	R.at<double>(0, 0) = cos(alpha);
	R.at<double>(0, 1) = -sin(alpha);
	R.at<double>(1, 0) = sin(alpha);
	R.at<double>(1, 1) = cos(alpha);
	R.at<double>(2, 2) = 1;

	G.at<double>(0, 0) = 1;
	G.at<double>(1, 1) = 1;
	G.at<double>(2, 2) = 1;
	G.at<double>(2, 0) = -1/f;

	Hp = G*R;
	Hp = Hp*T;

	xnew.x = centerx;
	xnew.y = centery;
	ptsTransform(Hp, xnew, xpnew);

	T2.at<double>(0, 0) = 1;
	T2.at<double>(0, 2) = centerx - xpnew.x;
	T2.at<double>(1, 1) = 1;
	T2.at<double>(1, 2) = centery - xpnew.y;
	T2.at<double>(2, 2) = 1;
	Hp = T2*Hp;

	SVD svd(P);
	Pinv = svd.vt.t()*Mat::diag(1./svd.w)*svd.u.t();
	M = Pp*Pinv;

	H0 = Hp*M;

	for(i = 0; i < num; i++)
	{
		ptsTransform(Hp, pt2[i], xpnew);
		ptsTransform(Hp, pt1[i], xnew);
		A.at<double>(i, 0) = xnew.x;
		A.at<double>(i, 1) = xnew.y;
		A.at<double>(i, 2) = 1;
		b.at<double>(i, 0) = xpnew.x;
	}
	solve(A, b, t, CV_SVD);

	Ha.at<double>(0, 0) = t.at<double>(0, 0);
	Ha.at<double>(0, 1) = t.at<double>(1, 0);
	Ha.at<double>(0, 2) = t.at<double>(2, 0);
	Ha.at<double>(1, 1) = 1;
	Ha.at<double>(2, 2) = 1;

	H = Ha*H0;

	Mat Hpinv = Hp.inv();
	Mat Hinv = H.inv();
	F = Hpinv.t()*F;
	F = F*Hinv;
}
*/


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
 *		interestPtsMatch - match interest points
 *		by NCC method for a row
 * Input:
 *		img1: left image 
 *		img2: right image
 *		p1: interest points in the left image
 *		num1: left interest points number 
 *		p2: interest points in the right image
 *		num2: right interest points number
 *		m1: matched points in the left image
 *		m2: matched points in the right image
 * Output:
 *		m1, m2
 *------------------------------------------------
 */

int interestPtsMatch(Mat img1, Mat img2, Point *p1, int num1, Point *p2, int num2, Point2d *m1, Point2d *m2)
{
	int i, j, idx = 0;
	double cur_val, max_val;
	int cur_x, cur_y, match_x, match_y;
	Scalar mean1, mean2;
	double *nccvals = new double[num1];
	int *matchedidx = new int[num1];
	Mat win1, win2, avg1Mat, avg2Mat, C1, C2, C;
	Size wsize = Size(WSIZE, WSIZE);
	int check = 0;

	Size imgSize = img1.size();
	int height = imgSize.height;
	int width = imgSize.width;

	for(i = 0; i < num1; i++)
	{
		max_val = -10000;
		cur_x = p1[i].x;
		cur_y = p1[i].y;
		m1[idx].x = (double)cur_x;
		m1[idx].y = (double)cur_y;
		
		// get neighborhood window of the left interest point
		win1 = getWindow(img1, Point(cur_x, cur_y), WSIZE);
		if(win1.size() != wsize) continue;
		for(j = 0; j < num2; j++)
		{
			if(pow(cur_x - p2[j].x, 2.0) > 400)
				continue;
			match_x = p2[j].x;
			match_y = p2[j].y;
			
			// get neighborhood window of the right interest point
			win2 = getWindow(img2, Point(match_x, match_y), WSIZE);
			if(win2.size() != wsize) continue;
			
			// calculate NCC value
			mean1 = mean(win1);
			mean2 = mean(win2);
			subtract(win1, mean1, avg1Mat);
			subtract(win2, mean2, avg2Mat);
			C = avg1Mat.mul(avg2Mat);
			C1 = avg1Mat.mul(avg1Mat);
			C2 = avg2Mat.mul(avg2Mat);

			cur_val = (double)sum(C).val[0]/sqrt(sum(C1).val[0]*sum(C2).val[0]);			

			// update the biggest NCC value
			if(cur_val > max_val)
			{
				max_val = cur_val;
				nccvals[idx] = cur_val;
				m2[idx].x = (double)match_x;
				m2[idx].y = (double)match_y;
				matchedidx[idx] = j;
			}
		}

		// check the matched points
		check = 0;
		for(j = 0; j < idx; j++)
		{
			if(matchedidx[j] == matchedidx[idx]){
				if(nccvals[j] < nccvals[idx]){
					nccvals[j] = nccvals[idx];
					m1[j].x = m1[idx].x;
					m1[j].y = m1[idx].y;
				}
				check = 1;
				break;
			}
		}
		if(check == 0)
			idx++;
	}

	// order the interest points on the row
	double tmpm1x, tmpm1y, tmpm2x, tmpm2y;
	for(i = 0; i < idx; i++)
	{
		for(j = 0; j < idx - i - 1; j++)
		{
			if(m1[j].x > m1[j+1].x){
				tmpm1x = m1[j].x;
				tmpm1y = m1[j].y;
				tmpm2x = m2[j].x;
				tmpm2y = m2[j].y;
				m1[j].x = m1[j+1].x;
				m1[j].y = m1[j+1].y;
				m2[j].x = m2[j+1].x;
				m2[j].y = m2[j+1].y;
				m1[j+1].x = tmpm1x;
				m1[j+1].y = tmpm1y;
				m2[j+1].x = tmpm2x;
				m2[j+1].y = tmpm2y;
			}
		}
	}
	
	delete nccvals;
	delete matchedidx;
	return idx;
}



int main()
{
	const char* img1_name = "img1.jpg";
	const char* img2_name = "img2.jpg";
	const int numpairs = 12;
	int i, j;

	// 12 points data
	double p1x[] = {162, 489, 58, 560, 334, 428, 330, 446, 62, 545, 458, 458};
	double p1y[] = {73, 92, 205, 240, 139, 143, 210, 217, 264, 296, 118, 129};
	double p2x[] = {197, 515, 71, 544, 345, 437, 320, 434, 74, 529, 477, 473};
	double p2y[] = {71, 98, 192, 253, 140, 148, 209, 222, 246, 308, 123, 134};

	vector<Point2d> pt1, pt2;

	for(i = 0; i < numpairs; i++)
	{
		pt1.push_back(Point2d(p1x[i], p1y[i]));
		pt2.push_back(Point2d(p2x[i], p2y[i]));
	}

	Mat img1 = imread(img1_name, CV_LOAD_IMAGE_COLOR);
	Mat img2 = imread(img2_name, CV_LOAD_IMAGE_COLOR);
	Size imgSize = img1.size();
	Mat imgOut(Size(imgSize.width*2, imgSize.height), CV_8UC3);
	// draw points pairs
	drawPairs(img1, img2, imgOut, pt1, pt2);
	imwrite("ptspairs.jpg", imgOut);

	// get fundamental matrix
	Mat F;
	F = getFundMat(numpairs, pt1, pt2);

	// get epipoles
	Mat e, ep;
	computeEpp(F, e, ep);

	// get canonical camera matrix
	Mat P, Pp;
	computeP(F, ep, P, Pp);

	// get initial 3d points
	vector<Point3d> X;
	reconstruct3D(numpairs, pt1, pt2, P, Pp, X);

	Point2d *p1 = &pt1[0];
	Point2d *p2 = &pt2[0];
	Point3d *Xt = &X[0];

	// LM non-linear optimization
	LMOptPara(numpairs, p1, p2, Pp, Xt, e, ep, F);
	
	Mat rimg1, rimg2;
	
	// rectify images based on F
	Mat H1, H2, H1inv, H2inv;
	stereoRectifyUncalibrated(pt1, pt2, F, imgSize, H1, H2, 3);
	H1inv = H1.inv();
	H2inv = H2.inv();

	// warp perspective images
	warpPerspective(img1, rimg1, H1, imgSize);
	warpPerspective(img2, rimg2, H2, imgSize);
	
	imwrite("rectified_img1.jpg", rimg1);
	imwrite("rectified_img2.jpg", rimg2);

	// canny edge detector
	Mat gimg1, gimg2, eimg1, eimg2;
	cvtColor(rimg1, gimg1, CV_BGR2GRAY);
	cvtColor(rimg2, gimg2, CV_BGR2GRAY);
	Canny(gimg1, eimg1, 20, 100);
	Canny(gimg2, eimg2, 20, 100);

	imwrite("edge_img1.jpg", eimg1);
	imwrite("edge_img2.jpg", eimg2);

	// find matching points and save 3d points
	Point interestPt1[MAX_INTEREST_PT];
	Point interestPt2[MAX_INTEREST_PT];
	Point2d matchedPt1[MAX_INTEREST_PT];
	Point2d matchedPt2[MAX_INTEREST_PT];
	Point2d transPt1[MAX_INTEREST_PT];
	Point2d transPt2[MAX_INTEREST_PT];
	vector<Point3d> reconstX;
	ofstream file;
	file.open("3dpoints.txt");
	int numpt1, numpt2, matchednum;
	bool nextline;

	Mat rimg(Size(imgSize.width*2, imgSize.height), CV_8UC3);
	rimg1.copyTo(rimg(Rect(0, 0, imgSize.width, imgSize.height)));
	rimg2.copyTo(rimg(Rect(imgSize.width, 0, imgSize.width, imgSize.height)));

	// cut the boundary edge
	for(i = 70; i < imgSize.height - 70; i++)
	{
		Point tmppt;
		nextline = false;
		numpt1 = 0;
		numpt2 = 0;
		// store detected edge points as interest points
		for(j = 80; j < imgSize.width - 80; j++)
		{
			if(eimg1.at<uchar>(i, j) == 255)
			{
				interestPt1[numpt1].x = j;
				interestPt1[numpt1].y = i;
				numpt1++;
			}
			if(eimg2.at<uchar>(i, j) == 255)
			{
				interestPt2[numpt2].x = j;
				interestPt2[numpt2].y = i;
				numpt2++;
			}
		}
		// match points for the current row
		matchednum = interestPtsMatch(gimg1, gimg2, interestPt1, numpt1, interestPt2, numpt2, matchedPt1, matchedPt2);
		for(j = 0; j < matchednum - 1; j++)
		{
			if(matchedPt2[j].x > matchedPt2[j+1].x){
				nextline = true;
				break;
			}
		}
		if(nextline)
			continue;
		// draw matching points
		for(j = 0; j < matchednum; j++)
		{
			circle(rimg, Point(matchedPt1[j].x, matchedPt1[j].y), 1, Scalar(255, 0, 0), 2, 8, 0);
			line(rimg, Point(matchedPt1[j].x, matchedPt1[j].y), Point(matchedPt2[j].x + imgSize.width, matchedPt2[j].y), Scalar(0, 255, 0));
			circle(rimg, Point(matchedPt2[j].x + imgSize.width, matchedPt2[j].y), 1, Scalar(255, 0, 0), 2, 8, 0);
		}
		// back project points from the rectified image to the original image
		for(j = 0; j < matchednum; j++)
		{
			ptsTransform(H1inv, matchedPt1[j], transPt1[j]);
			ptsTransform(H2inv, matchedPt2[j], transPt2[j]);
		}

		// triangulate 3d points
		vector<Point2d> vtransPt1(begin(transPt1), end(transPt1));
		vector<Point2d> vtransPt2(begin(transPt2), end(transPt2));
		reconstruct3D(matchednum, vtransPt1, vtransPt2, P, Pp, reconstX);
		// save 3d points
		for(j = 0; j < matchednum; j++)
		{
			file << reconstX[j].x << " " << reconstX[j].y << " " << reconstX[j].z << " \n";
		}
		vtransPt1.clear();
		vtransPt2.clear();
		reconstX.clear();
	}

	imshow("matched rimg", rimg);
	imwrite("matched_rimg.jpg", rimg);

	file.close();
	waitKey(0);

	return 0;
}