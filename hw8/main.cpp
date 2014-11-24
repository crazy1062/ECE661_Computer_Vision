#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv.hpp>
#include <levmar.h>

using namespace std;
using namespace cv;

#define CROW 10
#define CCOL 8
#define GRIDSIZE 36
#define RADIAL 0

typedef struct vhlines{
	vector<Vec2f> vlines;
	vector<Vec2f> hlines;
}VHLINE;

typedef struct ptsPairs{
	vector<vector<Point2f>> cSet;
	vector<Point2f> gSet;
}PTSPAIRS;


/*------------------------------------------------
 * Function: 
 *		readStringList - read the string list of 
 *		images
 * Input:
 *		filename: file name of list information 
 *		l: string container for read string list
 * Output:
 *		bool(read successfully or not)
 *------------------------------------------------
 */

bool readStringList(const string& filename, vector<string>& l)
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if(n.type() != FileNode::SEQ)
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for(; it != it_end; ++it)
        l.push_back((string)*it);
    return true;
}


/*--------------------------------------------------
 * Function: 
 *		drawHoughLines - draw Hough transformation
 *		lines 
 * Input:
 *		lines: container including parameters of
 *		Hough lines
 *		img: original image containing pattern
 * Output:
 *		Mat img_lines
 *--------------------------------------------------
 */

Mat drawHoughLines(vector<Vec2f> lines, const Mat& img)
{
	Mat img_lines = img.clone();

	for(size_t i = 0; i < lines.size(); i++)
	{
		double rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		line(img_lines, pt1, pt2, Scalar(0, 0, 255), 1, CV_AA);
	}

	return img_lines;
}


/*--------------------------------------------------
 * Function: 
 *		divideVHLines - divide vertical lines and
 *		horizontal lines
 * Input:
 *		lines: container including parameters of
 *		Hough lines
 * Output:
 *		VHLINE vh
 *--------------------------------------------------
 */

VHLINE divideVHLines(vector<Vec2f> lines)
{
	VHLINE vh;

	for(size_t i = 0; i < lines.size(); i++)
	{
		double theta = lines[i][1];
		theta = theta - double(CV_PI/2);
		// determine the vertical or horizontal lines
		if(abs(theta) < CV_PI/4){
			vh.hlines.push_back(lines[i]);
		}else{
			vh.vlines.push_back(lines[i]);
		}
	}

	return vh;
}


/*--------------------------------------------------
 * Function: 
 *		sortVLines - sort vertical lines by distance
 * Input:
 *		lhs: left hand side Vec2f
 *		rhs: right hand side Vec2f
 * Output:
 *		bool
 *--------------------------------------------------
 */

bool sortVLines(const Vec2f &lhs, const Vec2f &rhs)
{
	return abs(lhs[0]*cos(lhs[1])) < abs(rhs[0]*cos(rhs[1]));
}


/*----------------------------------------------------
 * Function: 
 *		sortHLines - sort horizontal lines by distance
 * Input:
 *		lhs: left hand side Vec2f 
 *		rhs: right hand side Vec2f
 * Output:
 *		bool
 *----------------------------------------------------
 */

bool sortHLines(const Vec2f &lhs, const Vec2f &rhs)
{
	return abs(lhs[0]*sin(lhs[1])) < abs(rhs[0]*sin(rhs[1]));
}


/*--------------------------------------------------
 * Function: 
 *		normHC - normalize point to homogeneous
 * Input:
 *		pt: point to be normalized
 * Output:
 *		Point3f pt
 *--------------------------------------------------
 */

void normHC(Point3f& pt)
{
	pt.x = pt.x/pt.z;
	pt.y = pt.y/pt.z;
	pt.z = 1.0;
}


/*--------------------------------------------------
 * Function: 
 *		getHoughHC - get homogeneous representation
 *		from a hough line
 * Input:
 *		houghl: hough line
 * Output:
 *		Point3f lineHC
 *--------------------------------------------------
 */

Point3f getHoughHC(Vec2f houghl)
{
	double rho = houghl[0], theta = houghl[1];
	Point3f pt0 = Point3f(rho*cos(theta), rho*sin(theta), 1.0);
	Point3f pt1 = Point3f(pt0.x + 100*sin(theta), pt0.y - 100*cos(theta), 1.0);
	Point3f lineHC = pt0.cross(pt1);
	return lineHC;
}


/*--------------------------------------------------
 * Function: 
 *		checkParallel - check if two lines are
 *		parallel
 * Input:
 *		l1: the first line
 *		l2: the second line
 * Output:
 *		int type
 *--------------------------------------------------
 */

int checkParallel(Point3f l1, Point3f l2)
{
	if(l1.y == 0 && l2.y == 0) return 1;
	if(l1.x == 0 && l2.x == 0) return 2;
	if(l1.x/l2.x == l1.y/l2.y) return 3;
	return 0;
}

/*--------------------------------------------------
 * Function: 
 *		groupLines - merge spurious lines to unique
 *		lines
 * Input:
 *		lines: container including parameters of
 *		Hough lines
 *		imgSize: image size
 *		minDist: threshold for the parallel lines'
 *		distance
 * Output:
 *		vector<Vec2f> unilines
 *--------------------------------------------------
 */

vector<Vec2f> groupLines(vector<Vec2f> lines, Size imgSize, double minDist)
{
	size_t lsize = lines.size();
	vector<Vec2f> unilines;
	unilines.push_back(lines[0]);
	for(size_t i = 1; i < lsize; i++)
	{
		Vec2f lastline = unilines.back();
		Vec2f curline = lines[i];
		Point3f unilineHC = getHoughHC(lastline);
		Point3f lineHC = getHoughHC(curline);
		Point3f crosspt = unilineHC.cross(lineHC);
		normHC(crosspt);
		// check if two lines intersect in the image
		if(crosspt.x >= 0 && crosspt.x < imgSize.width
			&& crosspt.y >= 0 && crosspt.y < imgSize.height){
			continue;		
		}
		
		// check if two lines are parallel
		int ptype = checkParallel(lineHC, unilineHC);
		if(ptype){
			double deltaDist = 0;
			if(ptype == 1){
				deltaDist = abs(lineHC.z/lineHC.x - unilineHC.z/unilineHC.x);
			}
			if(ptype == 2){
				deltaDist = abs(lineHC.z/lineHC.y - unilineHC.z/unilineHC.y);
			}
			if(ptype == 3){
				double k = lineHC.x/unilineHC.x;
				double rt = sqrt(lineHC.x*lineHC.x + lineHC.y*lineHC.y);
				deltaDist = abs(k*unilineHC.z - lineHC.z)/rt;
			}
			if(deltaDist <= minDist){
				continue;
			}
		}

		// the other cases are unique lines
		unilines.push_back(curline);
	}

	return unilines;
}


/*--------------------------------------------------
 * Function: 
 *		findCorner - find the detected corners
 * Input:
 *		img: original image
 *		idx: the index of the image list
 * Output:
 *		vector<Point2f> corners
 *--------------------------------------------------
 */

vector<Point2f> findCorner(const Mat& img, int idx)
{
	stringstream ss;
	ss << idx;
	string iname = ss.str();

	Mat img_gray, d_edges;
	Mat line1 = img.clone(), line2 = img.clone(), line3 = img.clone();
	Size imgSize = img.size();
	cvtColor(img, img_gray, CV_BGR2GRAY);
	//GaussianBlur(img_gray, img_gray, Size(3, 3), 0, 0);
	// edges detector
	Canny(img_gray, d_edges, 50, 300);
	imwrite("imgEdge" + iname + ".jpg", d_edges);

	vector<Vec2f> lines;
	// hough lines detector
	HoughLines(d_edges, lines, 1, CV_PI/180, 50);
	line1 = drawHoughLines(lines, img);
	imwrite("imgSlines" + iname + ".jpg", line1);

	VHLINE vh;
	vector<Vec2f> vlines, hlines;
	vh = divideVHLines(lines);
	
	// sort two directional lines
	sort(vh.vlines.begin(), vh.vlines.end(), sortVLines);
	sort(vh.hlines.begin(), vh.hlines.end(), sortHLines);

	// get unique lines
	vlines = groupLines(vh.vlines, imgSize, 15.0);
	hlines = groupLines(vh.hlines, imgSize, 15.0);

	line2 = drawHoughLines(vlines, img);
	line3 = drawHoughLines(hlines, line2);
	imwrite("imglines" + iname + ".jpg", line3);

	vector<Point2f> corners;

	// get two orthogonal lines' intersection as corners
	for(size_t i = 0; i < hlines.size(); i++)
	{
		Point3d hline = getHoughHC(hlines[i]);
		for(size_t j = 0; j < vlines.size(); j++)
		{
			Point3f vline = getHoughHC(vlines[j]);
			Point3f corner = vline.cross(hline);
			normHC(corner);
			corners.push_back(Point2d(corner.x, corner.y));
		}
	}

	// refine corners
	cornerSubPix(img_gray, corners, Size(15,15), Size(-1,-1),
		TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 30, 0.01));

	return corners;
}


/*--------------------------------------------------
 * Function: 
 *		getGroundTruth - get ground truth corners 
 * Input:
 *		gridLen: the grid length of the chessboard
 * Output:
 *		vector<Point2f> gTruth
 *--------------------------------------------------
 */

vector<Point2f> getGroundTruth(double gridLen)
{
	vector<Point2f> gTruth;
	for(int i = 0; i < CROW; i++)
	{
		for(int j = 0; j < CCOL; j++)
		{
			double x = j*gridLen;
			double y = i*gridLen;
			gTruth.push_back(Point2d(x, y));
		}
	}
	return gTruth;
}


/*--------------------------------------------------
 * Function: 
 *		getVij - get vij matrix
 * Input:
 *		i:index i 
 *		j:index j
 *		H:homography
 * Output:
 *		Mat vij
 *--------------------------------------------------
 */

Mat getVij(int i, int j, const Mat& H)
{
	Mat vij(6, 1, CV_64FC1, Scalar(1));
	vij.at<double>(0, 0) = H.at<double>(0, i - 1)*H.at<double>(0, j - 1);
	vij.at<double>(1, 0) = (H.at<double>(0, i - 1)*H.at<double>(1, j - 1) + H.at<double>(1, i - 1)*H.at<double>(0, j - 1));
	vij.at<double>(2, 0) = (H.at<double>(1, i - 1)*H.at<double>(1, j - 1));
	vij.at<double>(3, 0) = (H.at<double>(2, i - 1)*H.at<double>(0, j - 1) + H.at<double>(0, i - 1)*H.at<double>(2, j - 1));
	vij.at<double>(4, 0) = (H.at<double>(2, i - 1)*H.at<double>(1, j - 1) + H.at<double>(1, i - 1)*H.at<double>(2, j - 1));
	vij.at<double>(5, 0) = (H.at<double>(2, i - 1)*H.at<double>(2, j - 1));
	return vij;
}

/*--------------------------------------------------
 * Function: 
 *		solveIAC - solve mapped absolute conic
 * Input:
 *		Hset: container including H matrices
 * Output:
 *		Mat b
 *--------------------------------------------------
 */

Mat solveIAC(vector<Mat> Hset)
{
	int imgNum = Hset.size();
	Mat Vij(imgNum*2, 6, CV_64FC1, Scalar(0));
	Mat v12, v11, v22;

	for(int i = 0; i < imgNum; i++)
	{
		v12 = getVij(1, 2, Hset[i]);
		v11 = getVij(1, 1, Hset[i]);
		v22 = getVij(2, 2, Hset[i]);
		Vij.row(2*i) = v12.t();
		Vij.row(2*i + 1) = (v11 - v22).t();
	}

	Mat U, D, Vt;

	SVD::compute(Vij.t()*Vij, D, U, Vt, 0);

	cout << "U: " << U.size() << endl;
	cout << "D: " << D.size() << endl;
	cout << "Vt: " << Vt.size() << endl;

	Mat V = Vt.t();
	Mat b = V.col(5);

	return b;
}


/*--------------------------------------------------
 * Function: 
 *		calibInt - calibrate the intrinsics 
 * Input:
 *		para: contain all parameters
 *		Hset: H matrices
 *		K: camera matrix
 * Output:
 *		K, para
 *--------------------------------------------------
 */

void calibInt(double *para, vector<Mat> Hset, Mat& K)
{
	double w[6], lambda;
	Mat b = solveIAC(Hset);
	for(int i = 0; i < 6; i++)
	{
		w[i] = b.at<double>(i);
		cout << w[i] << endl;
	}

	para[4] = (w[1]*w[3] - w[0]*w[4])/(w[0]*w[2] - w[1]*w[1]);
	//cout << "y0: " << para[4] << endl;
	lambda = w[5] - (w[3]*w[3] + para[4]*(w[1]*w[3] - w[0]*w[4]))/w[0]; 
	//cout << "lambda: " << lambda << endl;
	para[0] = sqrt(lambda/w[0]);
	//cout << "alphax: " << para[0] << endl;
	para[1] = sqrt((lambda*w[0])/(w[0]*w[2] - w[1]*w[1]));
	//cout << "alphay: " << para[1] << endl;
	para[2] = -w[1]*para[0]*para[0]*para[1]/lambda;
	//cout << "s: " << para[2] << endl;
	para[3] = para[2]*para[4]/para[1] - w[3]*para[0]*para[0]/lambda;
	//cout << "x0: " << para[3] << endl;

	K = Mat::zeros(3, 3, CV_64FC1);
	K.at<double>(2, 2) = 1.0;
	K.at<double>(0, 0) = para[0];
	K.at<double>(0, 1) = para[2];
	K.at<double>(0, 2) = para[3];
	K.at<double>(1, 1) = para[1];
	K.at<double>(1, 2) = para[4];

}


/*--------------------------------------------------
 * Function: 
 *		rvecNorm - rotation vectors normalization
 * Input:
 *		r: rotation matrix
 *		norm: scale from normalization
 * Output:
 *		r
 *--------------------------------------------------
 */

void rvecNorm(Mat& r, double norm)
{
	r.at<double>(0) = r.at<double>(0)/norm;
	r.at<double>(1) = r.at<double>(1)/norm;
	r.at<double>(2) = r.at<double>(2)/norm;
}


/*--------------------------------------------------
 * Function: 
 *		calibExt - calibrate the extrinsics
 * Input:
 *		para: contain all parameters
 *		Hset: H matrices
 *		K: camera matrix
 * Output:
 *		para, K
 *--------------------------------------------------
 */

void calibExt(double *para, vector<Mat> Hset, Mat& K)
{
	int imgNum = Hset.size();
	Mat h1(3, 1, CV_64FC1, Scalar(0)), h2(3, 1, CV_64FC1, Scalar(0)), h3(3, 1, CV_64FC1, Scalar(0)); 
	Mat invK, r1, r2, r3, t;
	Mat Q(3, 3, CV_64FC1, Scalar(0));
	Mat U, V, D, Vt, Ut, R, Rnew(3, 1, CV_64FC1, Scalar(1));
	double norm;
	invK = K.inv();
	cout << "invK: " << invK << endl;
	
	for(int i = 0; i < imgNum; i++)
	{
		for(int j = 0; j < 3; j++){
			h1.at<double>(j, 0) = Hset[i].at<double>(j, 0);
			h2.at<double>(j, 0) = Hset[i].at<double>(j, 1);
			h3.at<double>(j, 0) = Hset[i].at<double>(j, 2);
		}
		//cout << "h1: " << h1 << endl;
		r1 = invK*h1;
		norm = sqrt(r1.dot(r1));
		rvecNorm(r1, norm);
		//cout << "r1: " << r1 << endl;
		r2 = invK*h2;
		rvecNorm(r2, norm);
		//cout << "r2: " << r2 << endl;
		r3 = r1.cross(r2);
		rvecNorm(r3, norm);
		//cout << "r3: " << r3 << endl;
		t = invK*h3;
		rvecNorm(t, norm);
		//cout << "t: " << t << endl;
		// rotation conditioning
		for(int k = 0; k < 3; k++)
		{
			Q.at<double>(k, 0) = r1.at<double>(k, 0);
			Q.at<double>(k, 1) = r2.at<double>(k, 0);
			Q.at<double>(k, 2) = r3.at<double>(k, 0);
		}
		//cout << "Q: " << Q << endl;
		SVD::compute(Q, D, U, Vt, 0);
		//cout << "D: " << D << endl;
		// rotaion conditioning
		R = U*Vt;
		//cout << "R: " << R.t()*R << endl;
		// convert to rodrigues representation
		Rodrigues(R, Rnew);
		//cout << "Rnew: " << Rnew << endl;
		para[7 + i*6] = Rnew.at<double>(0);
		para[7 + i*6 + 1] = Rnew.at<double>(1);
		para[7 + i*6 + 2] = Rnew.at<double>(2);
		para[7 + i*6 + 3] = t.at<double>(0);
		para[7 + i*6 + 4] = t.at<double>(1);
		para[7 + i*6 + 5] = t.at<double>(2);
	}
}


/*--------------------------------------------------
 * Function: 
 *		calibRad - calibrate radial distortion
 * Input:
 *		para: contain all parameters
 *		Hset: H matrices
 *		K: camera matrix
 *		cornersSet: container contains the set of 
 *		corners
 *		grounds: ground truth corners
 * Output:
 *		para, K
 *--------------------------------------------------
 */

void calibRad(double *para, vector<Mat> Hset, Mat& K, vector<vector<Point2f>> cornersSet, vector<Point2f> grounds)
{
	int imgNum = Hset.size();
	double u, v, x, y;
	double tmp1, tmp2, tmp3;
	double alpha = para[0];
	double beta = para[1];
	double gamma = para[2];
	double u0 = para[3];
	double v0 = para[4];
	Mat ptX(3, 1, CV_64FC1, Scalar(1));
	Mat ptx(3, 1, CV_64FC1, Scalar(1));
	Mat b(2, 1, CV_64FC1, Scalar(0));
	Mat D(2*80*20, 2, CV_64FC1, Scalar(0)), d(2*80*20, 1, CV_64FC1, Scalar(0));
	Mat R(3, 3, CV_64FC1, Scalar(1));
	vector<Point2f> corners;

	int index = 0;
	for(int i = 0; i < imgNum; i++)
	{
		for(int k = 0; k < 3; k++){
			R.at<double>(k, 0) = Hset[i].at<double>(k, 0);
			R.at<double>(k, 0) = Hset[i].at<double>(k, 1);
			R.at<double>(k, 0) = Hset[i].at<double>(k, 2);
		}
		corners = cornersSet[i];
		int csize = corners.size();
		for(int j = 0; j < csize; j++)
		{
			ptX.at<double>(0, 0) = (double)grounds[j].x;
			ptX.at<double>(1, 0) = (double)grounds[j].y;
			ptx = R*ptX;
			u = ptx.at<double>(0, 0)/ptx.at<double>(2, 0);
			v = ptx.at<double>(1, 0)/ptx.at<double>(2, 0);
			tmp1 = u - u0;
			tmp2 = v - v0;
			x = (u - u0)/alpha;
			y = (v - v0)/beta;
			d.at<double>(2*index, 0) = (double)corners[j].x - u;
			d.at<double>(2*index + 1, 0) = (double)corners[j].y - v;
			tmp3 = x*x + y*y;
			D.at<double>(2*index, 0) = tmp1*tmp3;
			D.at<double>(2*index, 1) = tmp1*tmp3*tmp3;
			D.at<double>(2*index + 1, 0) = tmp2*tmp3;
			D.at<double>(2*index + 1, 1) = tmp2*tmp3*tmp3;
			index++;
		}
	}
	solve(D, d, b, CV_SVD);
	para[5] = b.at<double>(0, 0);
	//cout << "k1: " << para[5] << endl;
	para[6] = b.at<double>(1, 0);
	//cout << "k2: " << para[6] << endl;
}


/*--------------------------------------------------
 * Function: 
 *		reprojectCorners - reproject corners to 
 *		images
 * Input:
 *		ground: ground truth point
 *		para: contain all parameters
 *		xtr[2]: reprojected point
 *		idx: index of image list
 *		isRadial: radial distortion bit
 * Output:
 *		para, xtr[2]
 *--------------------------------------------------
 */

void reprojectCorners(Point2f ground, double *para, double xtr[2], int idx, int isRadial)
{
	double k1 = para[5];
	double k2 = para[6];
	double u0 = para[3];
	double v0 = para[4];
	double alpha = para[0];
	double beta = para[1];
	double gamma = para[2];
	double u, v;
	double w_x = para[7 + 6*idx], w_y = para[7 + 6*idx + 1], w_z = para[7 + 6*idx + 2];
	double tx = para[7 + 6*idx + 3], ty = para[7 + 6*idx + 4], tz = para[7 + 6*idx + 5];
	Mat K(3, 3, CV_64FC1, Scalar(0));
	K.at<double>(2, 2) = 1.0;
	K.at<double>(0, 0) = alpha;
	K.at<double>(1, 1) = beta;
	K.at<double>(0, 1) = gamma;
	K.at<double>(0, 2) = u0;
	K.at<double>(1, 2) = v0;
	Mat R(3, 1, CV_64FC1, Scalar(1)), Rnew;
	R.at<double>(0, 0) = w_x;
	R.at<double>(1, 0) = w_y;
	R.at<double>(2, 0) = w_z;
	Rodrigues(R, Rnew);
	Rnew.at<double>(0, 2) = tx;
	Rnew.at<double>(1, 2) = ty;
	Rnew.at<double>(2, 2) = tz;
	Mat ptX(3, 1, CV_64FC1, Scalar(1)), ptx(3, 1, CV_64FC1, Scalar(1));
	ptX.at<double>(0, 0) = (double)ground.x;
	ptX.at<double>(1, 0) = (double)ground.y;
	// The computed homography: K*Rnew
	Mat H = Rnew;
	// projected corner point
	ptx = H*ptX;
	double x, y;
	x = ptx.at<double>(0, 0)/ptx.at<double>(2, 0);
	y = ptx.at<double>(1, 0)/ptx.at<double>(2, 0);
	u = u0 + alpha*x + gamma*y;
	v = v0 + beta*y;
	if(!isRadial){
		xtr[0] = x;
		xtr[1] = y;
	}else{
		double tmp = x*x + y*y;
		xtr[0] = u + (u - u0)*(k1*tmp + k2*tmp*tmp);
		xtr[1] = v + (v - v0)*(k1*tmp + k2*tmp*tmp);
	}
}


/*--------------------------------------------------
 * Function: 
 *		camCalibErrorFunc - calibration error 
 *		function for nonlinear optimization
 * Input:
 *		para: contain all parameters
 *		tran_x: reprojected point array
 *		m: parameter number
 *		n: data number
 *		adata: data for optimization
 * Output:
 *		para, tran_x
 *--------------------------------------------------
 */

static void camCalibErrorFunc(double *para, double *tran_x, int m, int n, void *adata)
{
	PTSPAIRS *pair = (PTSPAIRS *) adata;
	int num = pair->gSet.size()*pair->cSet.size();
	int gsize = pair->gSet.size();
	for(int i = 0; i < num; i++)
	{
		reprojectCorners(pair->gSet[i%gsize], para, tran_x + i*2, i, RADIAL);
	}
}


/*--------------------------------------------------
 * Function: 
 *		camCalibRefine - function for refinement
 * Input:
 *		para: contain all parameters
 *		Hset: H matrices
 *		ptsp: corner points pairs
 * Output:
 *		para
 *--------------------------------------------------
 */

void camCalibRefine(double *para, vector<Mat> Hset, PTSPAIRS ptsp)
{
	int imgNum = Hset.size(), ret;
	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];

	opts[0] = LM_INIT_MU;
	opts[1] = 1E-12;
	opts[2] = 1E-12;
	opts[3] = 1E-15;
	opts[4] = LM_DIFF_DELTA;
	void (*err)(double *p, double *hx, int m, int n, void *adata);
	int gsize = ptsp.gSet.size();
	int csize = ptsp.cSet.size();
	int LM_m = 7 + imgNum*6;
	int LM_n = 2*gsize*csize;
	double *ptx = (double *)malloc(LM_n*sizeof(double));

	for(int i = 0; i < csize; i++)
	{
		for(int j = 0; j < gsize; j++)
		{
			ptx[2*i] = ptsp.cSet[i][j].x;
			ptx[2*i + 1] = ptsp.cSet[i][j].y;
		}
	}

	err = camCalibErrorFunc;
	ret = dlevmar_dif(err, para, ptx, LM_m, LM_n, 1000, opts, info, NULL, NULL, &ptsp);
	printf("LM returned in %g iter, reason %g, sumsq %g [%g]\n", info[5], info[6], info[1], info[0]);

}


/*--------------------------------------------------
 * Function: 
 *		errorStat - statistics for errors
 * Input:
 *		grounds: ground truth corner points
 *		para: contain all parameters
 * Output:
 *		para
 *--------------------------------------------------
 */

double errorStat(vector<Point2f> grounds, double *para)
{
	int imgNum = 20;
	double error = 0;

	for(int i = 0; i < imgNum; i++)
	{
		double trans_pts[2], tmp1, tmp2, tmp3;

		for(int j = 0; j < grounds.size(); j++)
		{
			reprojectCorners(grounds[j], para, trans_pts, i, 0);
			tmp1 = (double)grounds[j].x - trans_pts[0];
			tmp2 = (double)grounds[j].y - trans_pts[1];
			tmp3 = tmp1*tmp1 + tmp2*tmp2;
			error += sqrt(tmp3);
		}
	}

	return error;
}

/*--------------------------------------------------
 * Function: 
 *		printMat - print the matrix
 *		
 * Input:
 *		para: contain all parameters
 *		idx: index of the image list
 * Output:
 *		para
 *--------------------------------------------------
 */

void printMat(double *para, int idx)
{
	int i = idx - 1;
	Mat K(3, 3, CV_64FC1, Scalar(0));
	K.at<double>(2, 2) = 1.0;
	K.at<double>(0, 0) = para[0];
	K.at<double>(1, 1) = para[1];
	K.at<double>(0, 1) = para[2];
	K.at<double>(0, 2) = para[3];
	K.at<double>(1, 2) = para[4];
	double k1, k2;
	k1 = para[5];
	k2 = para[6];
	Mat R(3, 3, CV_64FC1, Scalar(0)), rr(3, 1, CV_64FC1, Scalar(0)), t(3, 1, CV_64FC1, Scalar(0));
	rr.at<double>(0, 0) = para[7 + i*6];
	rr.at<double>(1, 0) = para[7 + i*6 + 1];
	rr.at<double>(2, 0) = para[7 + i*6 + 2];
	Rodrigues(rr, R);
	t.at<double>(0, 0) = para[7 + i*6 + 3];
	t.at<double>(1, 0) = para[7 + i*6 + 4];
	t.at<double>(2, 0) = para[7 + i*6 + 5];
	cout << "K: " << K << endl;
	cout << "R: " << R << endl;
	cout << "t: " << t << endl;
}


int main()
{
	char* imglist_name = "imglist2.xml";
	vector<string> imglist;
	bool ok = readStringList(imglist_name, imglist);
	if(!ok || imglist.empty())
	{
		cout << "can not open " << imglist_name << " or the string list is empty" << endl;
		return -1;
	}
	const int imgNum = imglist.size();

	/*-------------------------------------------------
	 * extracting corners process
	 *-------------------------------------------------
	 */


	PTSPAIRS ptsp;
	vector<Point2f> grounds = getGroundTruth(36);
	ptsp.gSet = grounds;

	vector<vector<Point2f>> cornersSet;
	vector<Mat> Hset;
	
	for(int i = 0; i < imgNum; i++)
	{
		Mat img = imread(imglist[i], CV_LOAD_IMAGE_COLOR);
		Size imgSize = img.size();
		Mat imgCorners = img.clone();
		//Mat imgP = img.clone();
		//Mat imgO;

		vector<Point2f> corners;
		
		// extract refined corners
		corners = findCorner(img, i);
		cout << "corners size: " << corners.size() << endl;
		// calculate homography
		Mat H = findHomography(grounds, corners, CV_RANSAC);
		Hset.push_back(H);
		//warpPerspective(imgP, imgO, H, Size(imgSize.width, imgSize.height));

		cornersSet.push_back(corners);
		
		// draw detected corners
		for(size_t j = 0; j < corners.size(); j++)
		{
			circle(imgCorners, corners[j], 2, Scalar(0, 255, 0), -1);
			stringstream ss;
			ss << j;
			string cname = ss.str();
			putText(imgCorners, cname, corners[j], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
		}
		
		stringstream ss2;
		ss2 << i;
		string iname = ss2.str();
		imwrite("imgCO" + iname + ".jpg", imgCorners);
		//imshow("imgP" + iname, imgO);
		corners.clear();
	}
	
	
	ptsp.cSet = cornersSet;
	cout << "cSet size: " << ptsp.cSet.size() << endl;
	cout << "gSet size: " << ptsp.gSet.size() << endl;

	/*------------------------------------------
	 * camera calibration process
	 *------------------------------------------
	 */

	Mat K;
	double *para;
	para = new double[7 + imgNum*6];

	calibInt(para, Hset, K);

	cout << "K: " << K << endl;

	calibExt(para, Hset, K);

	calibRad(para, Hset, K, cornersSet, grounds);

	double error = 0;
	error = errorStat(grounds, para);
	//cout << "error before refine: " << error << endl;

	int64 t = getTickCount();
	camCalibRefine(para, Hset, ptsp);
	t = getTickCount() - t;			//start count time for optimization
	printf("Time elapsed: %fms\n", t*1000/getTickFrequency());

	error = errorStat(grounds, para);
	//cout << "error after refine: " << error << endl;

	
	// draw reprojected corners and detected corners to compare
	for(int i = 0; i < imgNum; i++)
	{
		Mat img = imread(imglist[i], CV_LOAD_IMAGE_COLOR);
		Size imgSize = img.size();
		Mat imgCorners = img.clone();
		vector<Point2f> c_pts = cornersSet[i];

		double trans_pts[2];

		for(int j = 0; j < grounds.size(); j++)
		{
			// reproject corners
			reprojectCorners(grounds[j], para, trans_pts, i, RADIAL);

			circle(imgCorners, Point(c_pts[j].x, c_pts[j].y), 3, Scalar(0, 255, 0), -1);
			circle(imgCorners, Point((int)trans_pts[0], (int)trans_pts[1]), 2, Scalar(0, 0, 255), -1);
			stringstream ss;
			ss << j;
			string cname = ss.str();
			putText(imgCorners, cname, Point((int)trans_pts[0], (int)trans_pts[1]), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
		}

		stringstream ss2;
		ss2 << i;
		string iname = ss2.str();
		imwrite("reprojimg" + iname + ".jpg", imgCorners);
	}

	// print camera intrinsics and extrinsics
	printMat(para, 5);
	printMat(para, 10);
	printMat(para, 15);
	printMat(para, 20);

	char c = getchar();
	char d = getchar();

	waitKey(0);
	delete [] para;

	return 0;
}