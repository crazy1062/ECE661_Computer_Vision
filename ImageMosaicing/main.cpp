#include <opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <iterator>
#include <time.h>
#include <nonfree\nonfree.hpp>

using namespace std;
using namespace cv;

struct pointPair{
	Point2d p1;
	Point2d p2;
};

/*------------------------------------------------
 * Function: 
 *		convert2Homogeneous - convert point to 
 *		homogeneous representation
 * Input:
 *		pt: point which data type is 'Point2d'
 * Output:
 *		return Mat h
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
 *		homo: homogeneous point representation
 * Output:
 *		homo
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
 *		getRandomPair - get the random pairs of  
 *		correspondences, 4 < pair < 10
 * Input:
 *		pairNum: pair number of correpondences, 
 *		at least 4, at most 10
 *		pairs: computed pair correspondences 
 * Output:
 *		pairs
 *		return vector<pointPair> tmpPairs
 *------------------------------------------------
 */

vector<pointPair> getRandomPair(const int pairNum, vector<pointPair>& pairs)
{

	if((int)pairs.size() < pairNum) return pairs;

	vector<pointPair> tmpPairs;

	// establish random engine
	srand((unsigned int)time(0));

	for(int i = 0; i < pairNum; i++)
	{
		// get random index among the set of pairs
		int idx = rand()%pairs.size();
		tmpPairs.push_back(pairs[idx]);
		// tempororily erase the selected pair to avoid the duplicates
		pairs.erase(pairs.begin() + idx);
	}

	return tmpPairs;
}


/*------------------------------------------------
 * Function: 
 *		computeH - compute the homography based on  
 *		pair correspondences
 * Input:
 *		selPair: selected pair correspondences
 *		pairs: computed pair correspondences 
 * Output:
 *		pairs
 *		return vector<pointPair> tmpPairs
 *------------------------------------------------
 */

Mat computeH(vector<pointPair> selPair)
{
	int pair_size = selPair.size();
	Mat A(pair_size*2, 9, CV_64FC1);
	A = Scalar(0);
	for(int i = 0; i < pair_size; i++)
	{
		A.at<double>(2*i,3) = -selPair[i].p2.x;
		A.at<double>(2*i,4) = -selPair[i].p2.y;
		A.at<double>(2*i,5) = -1;
		A.at<double>(2*i,6) = selPair[i].p1.y*selPair[i].p2.x;
		A.at<double>(2*i,7) = selPair[i].p1.y*selPair[i].p2.y;
		A.at<double>(2*i,8) = selPair[i].p1.y;
		A.at<double>(2*i+1,0) = selPair[i].p2.x;
		A.at<double>(2*i+1,1) = selPair[i].p2.y;
		A.at<double>(2*i+1,2) = 1;
		A.at<double>(2*i+1,6) = -selPair[i].p1.x*selPair[i].p2.x;
		A.at<double>(2*i+1,7) = -selPair[i].p1.x*selPair[i].p2.y;
		A.at<double>(2*i+1,8) = -selPair[i].p1.x;
	}

	Mat U, D, Vt;

	// compute the eigenvector of A.t()*A
	SVD::compute(A.t()*A, D, U, Vt, 0);

	Mat V = Vt.t();
	// the smallest eigenvalue corresponding to the eigenvector
	Mat h = V.col(8);

	Mat H = Mat(h.t()).reshape(0, 3);

	return H;
}


/*------------------------------------------------
 * Function: 
 *		findCurrInliers - find the current inliers
 *		based on computed H and rest set of pairs
 * Input:
 *		pairs: the rest of the set of pairs
 *		H: homography computed by selected pairs
 *		disTres: geometric distance threshold
 *		currDistStd: current distance standard
 *		deviation
 * Output:
 *		currDistStd
 *		return vector<pointPair> inliers
 *------------------------------------------------
 */

vector<pointPair> findCurrInliers(vector<pointPair> pairs, Mat H, double distThres, double& currDistStd)
{
	vector<pointPair> inliers;
	vector<double> dist;
	Mat Hx;
	Mat x(3, 1, CV_64FC1);
	Mat xp(3, 1, CV_64FC1);
	double tmpDist = 0.0, distSum = 0.0;

	for(int i = 0; i < (int)pairs.size() - 1; i++)
	{
		xp.at<double>(0,0) = pairs[i].p1.x;
		xp.at<double>(1,0) = pairs[i].p1.y;
		xp.at<double>(2,0) = 1.0;
		x.at<double>(0,0) = pairs[i].p2.x;
		x.at<double>(1,0) = pairs[i].p2.y;
		x.at<double>(2,0) = 1.0;

		Hx = H*x;
		normHomogeneous(Hx);
		// compute the geometric distance between the estimated Hx and x'
		tmpDist = pow(Hx.at<double>(0,0) - xp.at<double>(0,0), 2.0) + 
			pow(Hx.at<double>(1,0) - xp.at<double>(1,0), 2.0);

		// only accept the pair smaller than decision threshold
		if(tmpDist < distThres)
		{
			inliers.push_back(pairs[i]);
			dist.push_back(tmpDist);
			distSum += tmpDist;
		}
	}
	double distMean = distSum / (double)dist.size();
	// calculate standard deviation
	if(dist.size() > 1){
		for(int i = 0; i < (int)dist.size(); i++)
		{
			currDistStd += pow(dist[i] - distMean, 2.0);
		}
		currDistStd /= (double)(dist.size() - 1);
	}

	return inliers;
}

/*------------------------------------------------
 * Function: 
 *		ransac - ransac algorithm to calculate the
 *		set of inlier correspondences
 * Input:
 *		pairs: all the corresponding pairs
 *		p: probability that at least one of the N
 *		trials will be free of outliers
 *		distThres: decision distance threshold
 * Output:
 *		return vector<pointPair> inliers
 *------------------------------------------------
 */


vector<pointPair> ransac(vector<pointPair> pairs, double p, double distThres)
{
	int size = (int)pairs.size();
	vector<pointPair> clonePairs = pairs;
	int N = numeric_limits<int>::max();
	int M = -1;
	//int N = (int)(log(1 - p)/log(1 - pow(1 - e, 5)));
	//int M = (int)((1 - e)*size);
	vector<pointPair> inliers, currentInliers, selectedPair;
	int Nupdate = 0;
	int numInliers;
	double distStd = 1000.0, currDistStd = 0.0;
	double e;
	Mat H;

	while(N > Nupdate)
	{
		if(!selectedPair.empty())
		{
			selectedPair.clear();
		}

		// shuffle the cloned pairs to maitain random selection
		random_shuffle(clonePairs.begin(), clonePairs.end());

		// randomly choose pairs for H estimation
		selectedPair = getRandomPair(5, clonePairs);

		H = computeH(selectedPair);

		currentInliers = findCurrInliers(clonePairs, H, distThres, currDistStd);

		numInliers = (int)currentInliers.size();
		
		// accept inliers with small standard deviation or the large amount
		if(numInliers > M || (numInliers == M && currDistStd < distStd))
		//if(numInliers >= M && currDistStd < distStd)
		{
			M = numInliers;

			if(!inliers.empty())
			{
				inliers.clear();
			}
			inliers = currentInliers;
			distStd = currDistStd;
		}

		// update the probability of false correspondences
		e = 1 - (double)numInliers/(double)size;
		// update the number of required trials
		N = (int)(log(1 - p)/log(1 - pow(1 - e, 5)));

		Nupdate++;
		cout << "Nupdate: " << Nupdate << endl;
		// reconstruct the clone pairs
		clonePairs.insert(clonePairs.end(), selectedPair.begin(), selectedPair.end());

	}

	return inliers;
	
}


/*------------------------------------------------
 * Function: 
 *		drawInliers - draw inliers on the image
 * Input:
 *		pairs: all corresponding pairs
 *		inliers: inliers of the set
 *		img1: left image
 *		img2: right image
 *		imgOut: output image
 * Output:
 *		imgOut
 *------------------------------------------------
 */

void drawInliers(vector<pointPair> pairs, vector<pointPair> inliers, const Mat& img1, const Mat& img2, Mat& imgOut)
{
	Size size = img1.size();
	img1.copyTo(imgOut(Rect(0, 0, size.width, size.height)));
	img2.copyTo(imgOut(Rect(size.width, 0, size.width, size.height)));
	// draw all the pairs firstly
	for(vector<pointPair>::iterator it = pairs.begin(); it != pairs.end(); it++)
	{
		circle(imgOut, it->p1, 4, Scalar(255, 0, 0));
		circle(imgOut, Point(it->p2.x + size.width, it->p2.y), 4, Scalar(255, 0, 0));
		line(imgOut, it->p1, Point(it->p2.x + size.width, it->p2.y), Scalar(255, 0, 0));
	}
	// draw the inliers to distinguish from the others
	for(vector<pointPair>::iterator it = inliers.begin(); it != inliers.end(); it++)
	{
		circle(imgOut, it->p1, 4, Scalar(0, 255, 0));
		circle(imgOut, Point(it->p2.x + size.width, it->p2.y), 4, Scalar(0, 255, 0));
		line(imgOut, it->p1, Point(it->p2.x + size.width, it->p2.y), Scalar(0, 255, 0));
	}
}


/*--------------------------------------------------
 * Function: 
 *		computeJacErr - compute Jacobian matrix 
 *		and Error matrix for nonlinear optimization
 * Input:
 *		inliers: inlier pairs
 *		h: homography vector, 9*1 size
 * Output:
 *		vector<Mat> JacErr
 *--------------------------------------------------
 */

vector<Mat> computeJacErr(vector<pointPair> inliers, Mat h)
{
	int size = (int)inliers.size();
	Mat E(2*size, 1, CV_64FC1);
	Mat J = Mat::zeros(2*size, 9, CV_64FC1);
	vector<Mat> JacErr;
	double xc, yc, wc, wc2;

	for(int i = 0; i < size; i++)
	{
		xc = h.at<double>(0,0)*inliers[i].p2.x + h.at<double>(1,0)*inliers[i].p2.y + h.at<double>(2,0);
		yc = h.at<double>(3,0)*inliers[i].p2.x + h.at<double>(4,0)*inliers[i].p2.y + h.at<double>(5,0);
		wc = h.at<double>(6,0)*inliers[i].p2.x + h.at<double>(7,0)*inliers[i].p2.y + h.at<double>(8,0);
		wc2 = wc*wc;

		E.at<double>(2*i,0) = inliers[i].p1.x - xc/wc;
		E.at<double>(2*i+1,0) = inliers[i].p1.y - yc/wc;

		J.at<double>(2*i,0) = -inliers[i].p2.x/wc;
		J.at<double>(2*i,1) = -inliers[i].p2.y/wc;
		J.at<double>(2*i,2) = -1.0/wc;
		J.at<double>(2*i,6) = (inliers[i].p2.x*xc)/wc2;
		J.at<double>(2*i,7) = (inliers[i].p2.y*xc)/wc2;
		J.at<double>(2*i,8) = xc/wc2;
		J.at<double>(2*i+1,3) = -inliers[i].p2.x/wc;
		J.at<double>(2*i+1,4) = -inliers[i].p2.y/wc;
		J.at<double>(2*i+1,5) = -1.0/wc;
		J.at<double>(2*i+1,6) = (inliers[i].p2.x*yc)/wc2;
		J.at<double>(2*i+1,7) = (inliers[i].p2.y*yc)/wc2;
		J.at<double>(2*i+1,8) = yc/wc2;
	}

	JacErr.push_back(J);
	JacErr.push_back(E);

	return JacErr;

}


/*------------------------------------------------
 * Function: 
 *		DoglegRefine - Dogleg method to refine the
 *		estimated homography vector
 * Input:
 *		H: homography of inliers
 *		inliers: inlier pairs
 *		error_thres: error threshold to ensure the
 *		precision
 * Output:
 *		Mat H
 *------------------------------------------------
 */


Mat DoglegRefine(Mat H, vector<pointPair> inliers, double error_thres)
{
	int size = (int)inliers.size();
	vector<Mat> JacErr;
	Mat J, E;
	// pk here means the h: homography vector
	Mat pk(9, 1, CV_64FC1), pknew(9, 1, CV_64FC1), pkdiff(9, 1, CV_64FC1), pktmp;
	// the output value of cost function
	double Cp, Cpnew, Cdiff;
	// delta pk in different cases(gradient descent or Gauss-Newton)
	Mat deltaGD(9, 1, CV_64FC1), deltaGN(9, 1, CV_64FC1), deltaGN_GD(9, 1, CV_64FC1);
	// norm of deltaGD and deltaGN
	double deltaGD_norm, deltaGN_norm;
	Mat J_t_error(9, 1, CV_64FC1), JJ_t_error(size*2, 1, CV_64FC1);
	double J_t_error_norm, JJ_t_error_norm;
	Mat Ident = Mat::eye(9, 9, CV_64FC1), tmp;
	pk = H.reshape(1, 9);
	// rk is the trust region
	double rk = 1.0, beta, a, b, c, pkdiff_norm, rhoDL;
	int i;

	for(i = 0; i < 200; i++)
	{
		JacErr = computeJacErr(inliers, pk);
		J = JacErr[0];
		E = JacErr[1];

		if(i > 0)
		{
			// Cp+1 = ||error||^2
			Cpnew = E.dot(E);
			// Cpdiff = Cp+1 - Cp
			Cdiff = Cpnew - Cp;
			//tmp = 2*pkdiff.t()*J.t()*E - pkdiff.t()*J.t()*J*pkdiff;
			tmp = 2*E.t()*J*pkdiff + pkdiff.t()*J.t()*J*pkdiff;
			// rhoDL = (Cp+1 - Cp)/(2*Error.t()*Jacob*deltap + deltap.t()*Jacob.t()*Jacob*deltap)
			rhoDL = Cdiff/tmp.at<double>(0);

			// update rhoDL
			if(rhoDL <= 0)
			{
				rk = rk*0.25;
				pktmp.copyTo(pk);
				JacErr = computeJacErr(inliers, pk);
				J = JacErr[0];
				E = JacErr[1];
			}
			else if(rhoDL <= 0.25)
			{
				rk = rk*0.25;
			}
			else if((rhoDL > 0.25)&&(rhoDL <= 0.75))
			{
				rk = rk*1.0;
			}
			else
			{
				rk = rk*2.0;
			}
		}

		// deltapGN and its norm
		// deltaGN = -(Jacob.t()*Jacob + mui*I)^-1*Jacob.t()*Error
		deltaGN = -(J.t()*J).inv()*J.t()*E;
		deltaGN_norm = sqrt(deltaGN.dot(deltaGN));
		// deltapGD and its norm
		J_t_error = J.t()*E;
		J_t_error_norm = J_t_error.dot(J_t_error);
		JJ_t_error = J*J.t()*E;
		JJ_t_error_norm = JJ_t_error.dot(JJ_t_error);
		// deltaGD = -||Jacob.t()*Error||/||Jacob*Jacob.t()*Error||*Jacob.t()*Error
		deltaGD = -J_t_error_norm/JJ_t_error_norm*J.t()*E;
		deltaGD_norm = sqrt(deltaGD.dot(deltaGD));

		// update trust region rk
		if(rk >= deltaGN_norm)
		{
			pknew = pk + deltaGN;
		}
		else if((rk < deltaGN_norm)&&(rk >= deltaGD_norm))
		{
			// ||deltapGD||^2 + beta^2*||deltapGN - deltapGD||^2 + 2*beta*deltapGD*(deltapGN - deltapGD) = rk^2
			deltaGN_GD = deltaGN - deltaGD;
			a = deltaGN_GD.dot(deltaGN_GD);
			b = 2*deltaGD.dot(deltaGN_GD);
			c = deltaGD.dot(deltaGD) - rk*rk;
			beta = (sqrt(b*b - 4*a*c) - b)/(2*a);
			pknew = pk + deltaGD + beta*deltaGN_GD;
		}
		else
		{
			pknew = pk + rk/deltaGD_norm*deltaGD;
		}

		pkdiff = pknew - pk;
		pkdiff_norm = pkdiff.dot(pkdiff);
		cout << "pkdiff_norm: " << pkdiff_norm << endl;
		cout << "iteration: " << i << endl;
		if(pkdiff_norm < error_thres) break;

		Cp = E.dot(E);
		cout << "Cp: " << Cp << endl;

		// backup pk
		pk.copyTo(pktmp);

		// update pk
		pknew.copyTo(pk);

	}

	H = Mat(pk.t()).reshape(1, 3);

	return H;

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
	height = (int)(ymax - ymin);
	width = (int)(height*aspect_ratio);
	//scale = height/(ymax - ymin);
	// we need the real world size, so the scale is 1.0 
	scale = 1.0;

	oFrame = Mat::zeros(Size(width, height), iFrame.type());

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

			if(resultx < 0 || resultx > frame_size.width - 1 || resulty < 0 || resulty > frame_size.height - 1){
				continue;
			}
				

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

	bound_oFrame.row(0) = bound_oFrame.row(0).mul(1/bound_oFrame.row(2));
	bound_oFrame.row(1) = bound_oFrame.row(1).mul(1/bound_oFrame.row(2));

	return bound_oFrame;

}


/*------------------------------------------------
 * Function: 
 *		mosaicking - image mosaicking based on a
 *		list of images and mutual homography
 * Input:
 *		Hset: mutual homography set
 *		imglist: image list
 * Output:
 *		Mat final_img
 *------------------------------------------------
 */

Mat mosaicking(vector<Mat> Hset, vector<Mat> imglist)
{
	int i, j, k, l;
	// compute the homography of each range image corresponding to the domain image
	vector<Mat> world_Hset1, world_Hset2;
	for(i = 0; i < (int)Hset.size()/2; i++)
	{
		Mat tmp = Mat::eye(3, 3, CV_64FC1);
		for(j = i; j < (int)Hset.size()/2; j++){
			tmp *= Hset[j];
		}
		world_Hset1.push_back(tmp);
	}
	for(i = (int)Hset.size() - 1; i >= (int)Hset.size()/2; i--)
	{
		Mat tmp = Mat::eye(3, 3, CV_64FC1);
		for(j = i; j >= (int)Hset.size()/2; j--){
			tmp *= Hset[j].inv();
		}
		world_Hset2.push_back(tmp);
	}
	reverse(world_Hset2.begin(), world_Hset2.end());
	world_Hset1.insert(world_Hset1.end(), world_Hset2.begin(), world_Hset2.end());


	Mat imgOut;
	vector<Mat> corrected_img;
	int size = (int)world_Hset1.size();
	Size frame_size = imglist[0].size();
	// bound of each range image in the world plane
	Mat bound_world;
	// the left top corner of the corrected image in the world plane
	vector<Point2d> lefttop;

	Mat domain_img;
	double xmin, xmax, ymin, ymax;
	double oxmin = 10000, oxmax = 0, oymin = 10000, oymax = 0;
	for(i = 0; i < size; i++){
		// store domain image which is in the middle of the set
		if(i == size/2){
			domain_img = imglist[i];
			lefttop.push_back(Point2d(0,0));
			corrected_img.push_back(domain_img);
		}
		if(i < size/2){
			bound_world = imageBackProj(imglist[i], imgOut, frame_size, world_Hset1[i].inv());
		}else{
			bound_world = imageBackProj(imglist[i+1], imgOut, frame_size, world_Hset1[i].inv());
		}
		// update the final image's bound
		minMaxLoc(bound_world.row(0), &xmin, &xmax, 0, 0);
		minMaxLoc(bound_world.row(1), &ymin, &ymax, 0, 0);
		if(xmin < oxmin) oxmin = xmin;
		if(xmax > oxmax) oxmax = xmax;
		if(ymin < oymin) oymin = ymin;
		if(ymax > oymax) oymax = ymax;
		// store left top corner of the corrected image
		lefttop.push_back(Point2d(xmin, ymin));
		// store the corrected image
		corrected_img.push_back(imgOut);
	}

	int width = (int)(oxmax - oxmin);
	int height = (int)(oymax - oymin);
	Mat final_img = Mat::zeros(height, width, imglist[0].type());

	// image mosaicking, concatenate all the corrected images
	for(l = 0; l < size + 1; l++)
	{
		for(i = 0; i < corrected_img[l].cols; i++)
		{
			for(j = 0; j < corrected_img[l].rows; j++)
			{
				Scalar intensity = corrected_img[l].at<uchar>(j,i);
				if(intensity.val[0] != 0){
					for(int k = 0; k < 3; ++k)
					{
						final_img.data[final_img.channels()*(final_img.cols*(j + (int)(lefttop[l].y - oymin)) + (i + (int)(lefttop[l].x - oxmin))) + k] = 
							corrected_img[l].data[corrected_img[l].channels()*(corrected_img[l].cols*j + i) + k];
					}
				}
			}
		}
	}

	return final_img;

}


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

int main()
{

	/*---------------------------------
	 * read the image list
	 *---------------------------------
	 */

	char* imglist_name = "imglist.xml";
	vector<string> imglist;
	bool ok = readStringList(imglist_name, imglist);
	if(!ok || imglist.empty())
    {
        cout << "can not open " << imglist_name << " or the string list is empty" << endl;
        return -1;
    }

	int listSize = (int)imglist.size();
	// minimum Hessian parameter for all the pairs
	int miniHessian[4] = {1100, 400, 10, 100};

	vector<Mat> Hset, imageSet;
	
	for(int i = 0; i < listSize - 1; i++)
	{

		Mat left = imread(imglist[i], CV_LOAD_IMAGE_COLOR);
		Mat right = imread(imglist[i+1], CV_LOAD_IMAGE_COLOR);

		cout << imglist[i] << endl;
		cout << imglist[i+1] << endl;

		Mat left_gray, right_gray;
		cvtColor(left, left_gray, CV_RGB2GRAY);
		cvtColor(right, right_gray, CV_RGB2GRAY);

		imageSet.push_back(left);
		if(i == listSize - 2) imageSet.push_back(right);

		/*-----------------------------------------------------
		 * SURF detector and store pair correspondences 
		 *-----------------------------------------------------
		 */


		SurfFeatureDetector detector(miniHessian[i]);

		vector<KeyPoint> keypoints_right, keypoints_left;

		detector.detect(right_gray, keypoints_right);
		detector.detect(left_gray, keypoints_left);

		SurfDescriptorExtractor extractor;

		// get descriptors
		Mat descriptors_right, descriptors_left;

		extractor.compute(right_gray, keypoints_right, descriptors_right);
		extractor.compute(left_gray, keypoints_left, descriptors_left);

		// initialize brute force matchers
		BFMatcher matcher(NORM_L2);

		vector<DMatch> matches;

		matcher.match(descriptors_left, descriptors_right, matches);

		double max_dist = 0, min_dist = 100;

		for(int i = 0; i < descriptors_right.rows; i++)
		{
			double dist = matches[i].distance;
			if(dist < min_dist) min_dist = dist;
			if(dist > max_dist) max_dist = dist;
		}

		printf("-- Max dist : %f \n", max_dist);
		printf("-- Min dist : %f \n", min_dist);

		// determine good matches
		vector<DMatch> good_matches;

		for(int i = 0; i < descriptors_right.rows; i++)
		{
			if(matches[i].distance < 3*min_dist)
			{
				good_matches.push_back(matches[i]);
			}
		}

		cout << "good matches size: " << good_matches.size() << endl;

		vector<Point2d> point_right;
		vector<Point2d> point_left;
		vector<pointPair> point_pair;

		for(int i = 0; i < (int)good_matches.size(); i++)
		{
			pointPair p;
			p.p1 = keypoints_left[good_matches[i].queryIdx].pt;
			p.p2 = keypoints_right[good_matches[i].trainIdx].pt;
			point_pair.push_back(p);
		}

		cout << "point pair size: " << point_pair.size() << endl;

		Mat img_matches;

		// draw good matches
		drawMatches(left, keypoints_left, right, keypoints_right,
			good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		imshow(imglist[i] + imglist[i+1], img_matches);
		imwrite(imglist[i] + imglist[i+1] + ".jpg", img_matches);


		/*-------------------------------------------------------------
		 * RANSAC algorithm
		 *-------------------------------------------------------------
		 */

		
		vector<pointPair> inliers;

		inliers = ransac(point_pair, 0.99, 5.0);
		cout << "inliers size: " << inliers.size() << endl;

		Size img_size = left.size();
		Mat dst(Size(img_size.width*2, img_size.height), left.type());

		drawInliers(point_pair, inliers, left, right, dst);

		imshow(imglist[i] + imglist[i+1] + "inliers", dst);
		imwrite(imglist[i] + imglist[i+1] + "inliers.jpg", dst);

		/*-------------------------------------------------------------
		 * DogLeg nonlinear optimization algorithm
		 *-------------------------------------------------------------
		 */

		
		Mat H = computeH(inliers);

		cout << "ransac H: " << H << endl;

		H = DoglegRefine(H, inliers, 1e-20);

		cout << "dogleg H: " << H << endl;

		Hset.push_back(H);

	}
	
	/*--------------------------------------------
	 * image mosaicking
	 *--------------------------------------------
	 */

	Mat final_img = mosaicking(Hset, imageSet);

	imshow("final image", final_img);
	imwrite("final_img.jpg", final_img);

	waitKey(0);

	return 0;
}

