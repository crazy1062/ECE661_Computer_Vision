#include<opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;


/*------------------------------------------------------------------
 * Function: 
 *		 initMat - initialze matrices needed in the linear algebraic
 *		 systems
 * Input: 
 *		 Point2f image[7] - 8 object coordinates in original plane
 *		 Point2f frame[7] - 8 object coordinates in projected plane
 *		 Mat Avec - matrix left multiply the H
 *		 Mat bvec - matrix contain 16 coordinates in projected plane
 * Output:
 *		 Avec, bvec
 *------------------------------------------------------------------
 */


void initMat(const Point2f image[7], const Point2f frame[7], Mat& Avec, Mat& bvec)
{
	 const int size = 8;
	 for(int i = 0; i < size; ++i)
	 {
		 for(int k = 0; k < 2; ++k)
		 {
			 if(k == 0){
				 Avec.at<float>(2*i+k,0) = image[i].x;
				 Avec.at<float>(2*i+k,1) = image[i].y;
				 Avec.at<float>(2*i+k,2) = 1.f;
				 Avec.at<float>(2*i+k,3) = 0.f;
				 Avec.at<float>(2*i+k,4) = 0.f;
				 Avec.at<float>(2*i+k,5) = 0.f;
				 Avec.at<float>(2*i+k,6) = -image[i].x*frame[i].x;
				 Avec.at<float>(2*i+k,7) = -image[i].y*frame[i].x;
				 bvec.at<float>(2*i+k,0) = frame[i].x;
			 }else{
				 Avec.at<float>(2*i+k,0) = 0.f;
				 Avec.at<float>(2*i+k,1) = 0.f;
				 Avec.at<float>(2*i+k,2) = 0.f;
				 Avec.at<float>(2*i+k,3) = image[i].x;
				 Avec.at<float>(2*i+k,4) = image[i].y;
				 Avec.at<float>(2*i+k,5) = 1.f;
				 Avec.at<float>(2*i+k,6) = -image[i].x*frame[i].y;
				 Avec.at<float>(2*i+k,7) = -image[i].y*frame[i].y;
				 bvec.at<float>(2*i+k,0) = frame[i].y;
			 }
		 }
	 }
}


/*----------------------------------------------------------------------
 * Function: 
 *		 World2Frame - project the world actual image to the destination
 *		 frame
 * Input: 
 *		 Mat frame - the destination frame matrix 
 *		 Mat world - the world actual image matrix
 *		 Size frame_size - the size of the frame matrix
 *		 Size world_size - the size of the world matrix
 *		 Mat hVector - H transformation matrix
 * Ouput:
 *		 frame
 *----------------------------------------------------------------------
 */


void World2Frame(Mat& frame, const Mat world, const Size frame_size, const Size world_size, const Mat hVector)
{
	const int world_w = world_size.width, world_h = world_size.height;
	const int frame_w = frame_size.width, frame_h = frame_size.height;
	Point2f tmpPoint;
	Mat tmpMat, tmpResult;
	
	for(int i = 0; i < world_w; ++i)
	{
		tmpPoint.x = (float)i;
		for(int j = 0; j < world_h; ++j)
		{
			tmpPoint.y = (float)j;
			tmpMat = Mat(tmpPoint);	
			tmpMat.push_back(1.f);	// construct the homogeneous coordinate
			tmpResult = hVector*tmpMat;		// calculated result homogeneous point in the frame image
			int resultx = ceil(tmpResult.at<float>(0,0)/tmpResult.at<float>(2,0));	// get the result x coordinate in the frame image
			int resulty = ceil(tmpResult.at<float>(1,0)/tmpResult.at<float>(2,0));	// get the result y coordinate in the frame image

			// assign R, G, B value in the world image to the corresponding coordinate in the frame image
			for(int k = 0; k < 3; ++k)
			{
				frame.data[frame.channels()*(frame.cols*resulty + resultx) + k] = world.data[world.channels()*(world.cols*j + i) + k];
			}
		}
	}
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
	float xmin, xmax, ymin, ymax, xtmp, ytmp;
	float scale, aspect_ratio;
	int height, width;
	Point2f tmpPoint;
	Mat tmpResult, tmpMat;
	Mat hinv = hVector.inv(DECOMP_SVD);

	Mat bound_iFrame(3, 4, CV_32F);
	Mat bound_oFrame(3, 4, CV_32F);

	// construct four points of input frame
	bound_iFrame.at<float>(0,0) = 0;
	bound_iFrame.at<float>(1,0) = 0;
	bound_iFrame.at<float>(2,0) = 1.f;
	bound_iFrame.at<float>(0,1) = (float)(frame_size.width - 1);
	bound_iFrame.at<float>(1,1) = 0;
	bound_iFrame.at<float>(2,1) = 1.f;
	bound_iFrame.at<float>(0,2) = (float)(frame_size.width - 1);
	bound_iFrame.at<float>(1,2) = (float)(frame_size.height - 1);
	bound_iFrame.at<float>(2,2) = 1;
	bound_iFrame.at<float>(0,3) = 0;
	bound_iFrame.at<float>(1,3) = (float)(frame_size.height - 1);
	bound_iFrame.at<float>(2,3) = 1;
	
	bound_oFrame = hVector*bound_iFrame;

	// calculate the bound values of the projected vertices
	xmin = ymin = 1e10; xmax = ymax = 0;
	for(int i = 0; i < 4; ++i)
	{
		xtmp = bound_oFrame.at<float>(0, i)/bound_oFrame.at<float>(2, i);
		ytmp = bound_oFrame.at<float>(1, i)/bound_oFrame.at<float>(2, i);
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
	width = height*aspect_ratio;
	scale = height/(ymax - ymin);

	oFrame = Mat(Size(width, height), iFrame.type());

	// set the new H inverse with offsets
	hinv.at<float>(0,2) = scale*(xmin*hinv.at<float>(0,0) + ymin*hinv.at<float>(0,1) + hinv.at<float>(0,2));
	hinv.at<float>(1,2) = scale*(xmin*hinv.at<float>(1,0) + ymin*hinv.at<float>(1,1) + hinv.at<float>(1,2));
	hinv.at<float>(2,2) = scale*(xmin*hinv.at<float>(2,0) + ymin*hinv.at<float>(2,1) + hinv.at<float>(2,2));

	for(int i = 0; i < width; ++i)
	{
		tmpPoint.x = (float)i;
		for(int j = 0; j < height; ++j)
		{
			tmpPoint.y = (float)j;
			tmpMat = Mat(tmpPoint);
			tmpMat.push_back(1.f);
			tmpResult = hinv*tmpMat;
			float resultx = tmpResult.at<float>(0,0)/tmpResult.at<float>(2,0);
			float resulty = tmpResult.at<float>(1,0)/tmpResult.at<float>(2,0);

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


/*------------------------------------------------------------------------
 * Function: 
 *		 merge_img - merge the world image to the frame
 * Input: 
 *		 Mat frame - the target frame image 
 *		 Mat world - the world frame image
 *		 Mat hinv - new H inverse matrix to calculate the location of the
 *		 resized world image
 *		 float topleftx - the x coordinate of the topleft target point
 *		 float toplefty - the y coordinate of the topleft target point
 *		 float bottomrightx - the x coordinate of the bottomright target
 *		 point
 *		 float bottomrighty - the y coordinate of the bottomright target
 *		 point
 * Ouput:
 *		 frame
 *------------------------------------------------------------------------
 */


void merge_img(Mat& frame, Mat world, Mat hinv, float topleftx, float toplefty, float bottomrightx, float bottomrighty)
{
	// construct topleft and bottomright homogeneous coordinates
	Point2f topleft(topleftx, toplefty);
	Point2f bottomright(bottomrightx, bottomrighty);
	Mat topl = Mat(topleft);
	Mat btmr = Mat(bottomright);
	topl.push_back(1.f);
	btmr.push_back(1.f);

	// calculate the new topleft and bottomright coordinates
	Mat newH = hinv.inv(DECOMP_SVD);
	Mat Ptopl = newH*topl;
	Mat Pbtmr = newH*btmr;
	int newToplx = (int)Ptopl.at<float>(0,0)/Ptopl.at<float>(2,0);
	int newToply = (int)Ptopl.at<float>(1,0)/Ptopl.at<float>(2,0);

	int newBtmrx = (int)Pbtmr.at<float>(0,0)/Pbtmr.at<float>(2,0);
	int newBtmry = (int)Pbtmr.at<float>(1,0)/Pbtmr.at<float>(2,0);

	// construct the target frame size
	Size newWorldSize = Size(newBtmrx - newToplx, newBtmry - newToply);

	//resize the world image so that the target frame fits the image
	resize(world, world, newWorldSize);

	//merge two images at the designed location
	Rect roi(Point(newToplx, newToply), newWorldSize);
	world.copyTo(frame(roi));

	cout << "Ptopl: " << newToplx << " " << newToply << endl;
	cout << "Pbtmr: " << newBtmrx << " " << newBtmry << endl;
	
}

int main()
{
	/*-------------------------------
	 * Task 1: world to frame process
	 *-------------------------------
	 */

	Point2f image[7], world[7];
	image[0] = Point2f(0.f, 0.f);
	image[1] = Point2f(250.f, 0.f);
	image[2] = Point2f(500.f, 0.f);
	image[3] = Point2f(500.f, 254.f);
	image[4] = Point2f(500.f, 508.f);
	image[5] = Point2f(250.f, 508.f);
	image[6] = Point2f(0.f, 508.f);
	image[7] = Point2f(0.f, 254.f);
	world[0] = Point2f(188.f, 152.f);
	world[1] = Point2f(267.f, 163.f);
	world[2] = Point2f(346.f, 174.f);
	world[3] = Point2f(345.f, 302.f);
	world[4] = Point2f(345.f, 431.f);
	world[5] = Point2f(265.f, 446.f);
	world[6] = Point2f(186.f, 461.f);
	world[7] = Point2f(187.f, 306.f);

	Mat hvec;
	Mat Avec = Mat(16, 8, CV_32F);
	Mat bvec = Mat(16, 1, CV_32F);

	initMat(image, world, Avec, bvec);
	solve(Avec, bvec, hvec, DECOMP_SVD);
	hvec.push_back(1.f);
	hvec = hvec.reshape(0, 3);

	char* frame_name = "Frame.jpg";
	char* image_name = "Audrey.jpg";
	Mat frame_img = imread(frame_name, CV_LOAD_IMAGE_COLOR);
	Mat world_img = imread(image_name, CV_LOAD_IMAGE_COLOR);
	Mat frame_img1 = frame_img.clone();
	Size frame_size = frame_img.size();
	Size world_size = world_img.size();

	World2Frame(frame_img1, world_img, frame_size, world_size, hvec);
	
	/*-------------------------------
	 * Task 2: frame to world process
	 *-------------------------------
	 */

	image[0] = Point2f(486.f, 198.f);
	image[1] = Point2f(526.f, 203.f);
	image[2] = Point2f(567.f, 209.f);
	image[3] = Point2f(567.f, 307.f);
	image[4] = Point2f(567.f, 406.f);
	image[5] = Point2f(526.f, 413.f);
	image[6] = Point2f(486.f, 421.f);
	image[7] = Point2f(486.f, 309.f);
	world[0] = Point2f(251.f, 28.f);
	world[1] = Point2f(501.f, 28.f);
	world[2] = Point2f(751.f, 28.f);
	world[3] = Point2f(751.f, 282.f);
	world[4] = Point2f(751.f, 536.f);
	world[5] = Point2f(501.f, 536.f);
	world[6] = Point2f(251.f, 536.f);
	world[7] = Point2f(251.f, 282.f);

	initMat(image, world, Avec, bvec);
	solve(Avec, bvec, hvec, DECOMP_SVD);
	hvec.push_back(1.f);
	hvec = hvec.reshape(0, 3);

	Mat frame_img2;
	frame_img2 = Mat(frame_size.height, frame_size.width, frame_img.type());

	Mat newHinv = imageBackProj(frame_img, frame_img2, frame_size, hvec);
	merge_img(frame_img2, world_img, newHinv, 486.f, 198.f, 567.f, 406.f);

	/*-------------------------------------------------------------
	 * Task 3: reproject based on the real size of the target frame
	 *-------------------------------------------------------------
	 */

	image[0] = Point2f(188.f, 152.f);
	image[1] = Point2f(267.f, 163.f);
	image[2] = Point2f(346.f, 174.f);
	image[3] = Point2f(346.f, 302.f);
	image[4] = Point2f(346.f, 431.f);
	image[5] = Point2f(265.f, 446.f);
	image[6] = Point2f(186.f, 461.f);
	image[7] = Point2f(187.f, 306.f);
	world[0] = Point2f(170.f, 174.f);
	world[1] = Point2f(258.f, 174.f);
	world[2] = Point2f(346.f, 174.f);
	world[3] = Point2f(346.f, 302.f);
	world[4] = Point2f(346.f, 430.f);
	world[5] = Point2f(258.f, 430.f);
	world[6] = Point2f(170.f, 430.f);
	world[7] = Point2f(170.f, 302.f);

	initMat(image, world, Avec, bvec);
	solve(Avec, bvec, hvec, DECOMP_SVD);
	hvec.push_back(1.f);
	hvec = hvec.reshape(0, 3);

	Mat frame_img3;
	frame_img3 = Mat(frame_size.height, frame_size.width, frame_img.type());

	imageBackProj(frame_img1, frame_img3, frame_size, hvec);

	image[0] = Point2f(282.f, 252.f);
	image[1] = Point2f(323.f, 252.f);
	image[2] = Point2f(365.f, 252.f);
	image[3] = Point2f(365.f, 294.f);
	image[4] = Point2f(365.f, 336.f);
	image[5] = Point2f(323.f, 336.f);
	image[6] = Point2f(282.f, 336.f);
	image[7] = Point2f(282.f, 294.f);
	world[0] = Point2f(282.f, 252.f);
	world[1] = Point2f(310.f, 252.f);
	world[2] = Point2f(338.f, 252.f);
	world[3] = Point2f(338.f, 294.f);
	world[4] = Point2f(338.f, 336.f);
	world[5] = Point2f(310.f, 336.f);
	world[6] = Point2f(282.f, 336.f);
	world[7] = Point2f(282.f, 294.f);

	initMat(image, world, Avec, bvec);
	solve(Avec, bvec, hvec, DECOMP_SVD);
	hvec.push_back(1.f);
	hvec = hvec.reshape(0, 3);

	Mat frame_img4;

	imageBackProj(frame_img2, frame_img4, frame_img2.size(), hvec);

	
	namedWindow("iframe", WINDOW_AUTOSIZE);
	namedWindow("iframe2", WINDOW_AUTOSIZE);
	namedWindow("iframe3", WINDOW_AUTOSIZE);
	namedWindow("iframe4", WINDOW_AUTOSIZE);
	imshow("iframe", frame_img1);
	imshow("iframe2", frame_img2);
	imshow("iframe3", frame_img3);
	imshow("iframe4", frame_img4);
	


	/*-----------------------------------------------------------------------
	 * Own pair of images, apply the similar steps above to see results
	 *-----------------------------------------------------------------------
	 */


	char* frame2_name = "Mirror.jpg";
	char* image2_name = "Attractor.jpg";
	Mat frame2_img = imread(frame2_name, CV_LOAD_IMAGE_COLOR);
	Mat world2_img = imread(image2_name, CV_LOAD_IMAGE_COLOR);
	Mat frame2_img1 = frame2_img.clone();
	Size frame2_size = frame2_img.size();
	Size world2_size = world2_img.size();

	image[0] = Point2f(0.f, 0.f);
	image[1] = Point2f(250.f, 0.f);
	image[2] = Point2f(500.f, 0.f);
	image[3] = Point2f(500.f, 250.f);
	image[4] = Point2f(500.f, 500.f);
	image[5] = Point2f(250.f, 500.f);
	image[6] = Point2f(0.f, 500.f);
	image[7] = Point2f(0.f, 250.f);
	world[0] = Point2f(169.f, 130.f);
	world[1] = Point2f(296.f, 117.f);
	world[2] = Point2f(424.f, 105.f);
	world[3] = Point2f(423.f, 335.f);
	world[4] = Point2f(422.f, 565.f);
	world[5] = Point2f(300.f, 547.f);
	world[6] = Point2f(178.f, 529.f);
	world[7] = Point2f(173.f, 329.f);

	initMat(image, world, Avec, bvec);
	solve(Avec, bvec, hvec, DECOMP_SVD);
	hvec.push_back(1.f);
	hvec = hvec.reshape(0, 3);

	World2Frame(frame2_img1, world2_img, frame2_size, world2_size, hvec);


	image[0] = Point2f(169.f, 130.f);
	image[1] = Point2f(296.f, 117.f);
	image[2] = Point2f(424.f, 105.f);
	image[3] = Point2f(423.f, 335.f);
	image[4] = Point2f(422.f, 565.f);
	image[5] = Point2f(300.f, 547.f);
	image[6] = Point2f(178.f, 529.f);
	image[7] = Point2f(173.f, 329.f);
	world[0] = Point2f(250.f, 125.f);
	world[1] = Point2f(500.f, 125.f);
	world[2] = Point2f(750.f, 125.f);
	world[3] = Point2f(750.f, 375.f);
	world[4] = Point2f(750.f, 625.f);
	world[5] = Point2f(500.f, 625.f);
	world[6] = Point2f(250.f, 625.f);
	world[7] = Point2f(250.f, 375.f);

	initMat(image, world, Avec, bvec);
	solve(Avec, bvec, hvec, DECOMP_SVD);
	hvec.push_back(1.f);
	hvec = hvec.reshape(0, 3);

	Mat frame2_img2;
	frame2_img2 = Mat(frame2_size.height, frame2_size.width, frame2_img.type());

	newHinv = imageBackProj(frame2_img, frame2_img2, frame2_size, hvec);
	merge_img(frame2_img2, world2_img, newHinv, 169.f, 130.f, 422.f, 565.f);

	image[0] = Point2f(169.f, 130.f);
	image[1] = Point2f(296.f, 117.f);
	image[2] = Point2f(424.f, 105.f);
	image[3] = Point2f(423.f, 335.f);
	image[4] = Point2f(422.f, 565.f);
	image[5] = Point2f(300.f, 547.f);
	image[6] = Point2f(178.f, 529.f);
	image[7] = Point2f(173.f, 329.f);
	world[0] = Point2f(169.f, 130.f);
	world[1] = Point2f(316.f, 130.f);
	world[2] = Point2f(463.f, 130.f);
	world[3] = Point2f(463.f, 329.f);
	world[4] = Point2f(463.f, 529.f);
	world[5] = Point2f(316.f, 529.f);
	world[6] = Point2f(169.f, 529.f);
	world[7] = Point2f(169.f, 329.f);

	initMat(image, world, Avec, bvec);
	solve(Avec, bvec, hvec, DECOMP_SVD);
	hvec.push_back(1.f);
	hvec = hvec.reshape(0, 3);

	Mat frame2_img3;
	frame2_img3 = Mat(frame2_size.height, frame2_size.width, frame2_img.type());

	imageBackProj(frame2_img1, frame2_img3, frame2_size, hvec);

	image[0] = Point2f(338.f, 131.f);
	image[1] = Point2f(526.f, 131.f);
	image[2] = Point2f(714.f, 131.f);
	image[3] = Point2f(714.f, 318.f);
	image[4] = Point2f(714.f, 506.f);
	image[5] = Point2f(526.f, 506.f);
	image[6] = Point2f(338.f, 506.f);
	image[7] = Point2f(338.f, 318.f);
	world[0] = Point2f(338.f, 131.f);
	world[1] = Point2f(476.f, 131.f);
	world[2] = Point2f(614.f, 131.f);
	world[3] = Point2f(614.f, 318.f);
	world[4] = Point2f(614.f, 506.f);
	world[5] = Point2f(476.f, 506.f);
	world[6] = Point2f(338.f, 506.f);
	world[7] = Point2f(338.f, 318.f);

	initMat(image, world, Avec, bvec);
	solve(Avec, bvec, hvec, DECOMP_SVD);
	hvec.push_back(1.f);
	hvec = hvec.reshape(0, 3);

	Mat frame2_img4;

	imageBackProj(frame2_img2, frame2_img4, frame2_img2.size(), hvec);

	namedWindow("iframe5", WINDOW_AUTOSIZE);
	namedWindow("iframe6", WINDOW_AUTOSIZE);
	namedWindow("iframe7", WINDOW_AUTOSIZE);
	namedWindow("iframe8", WINDOW_AUTOSIZE);
	imshow("iframe5", frame2_img1);
	imshow("iframe6", frame2_img2);
	imshow("iframe7", frame2_img3);
	imshow("iframe8", frame2_img4);

	imwrite("frame1_img1.jpg", frame_img1);
	imwrite("frame1_img2.jpg", frame_img2);
	imwrite("frame1_img3.jpg", frame_img3);
	imwrite("frame1_img4.jpg", frame_img4);
	imwrite("frame2_img1.jpg", frame2_img1);
	imwrite("frame2_img2.jpg", frame2_img2);
	imwrite("frame2_img3.jpg", frame2_img3);
	imwrite("frame2_img4.jpg", frame2_img4);

	waitKey(0);

	return 0;
}