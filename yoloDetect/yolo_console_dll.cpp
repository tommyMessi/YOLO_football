#include <iostream>
#include <iomanip> 
#include <string>
#include <vector>
#include <fstream>

#define OPENCV

#include "M-DET.h"	

#ifdef OPENCV
#include <opencv2/opencv.hpp>			// C++


#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../tracking/kcftracker.hpp"

void draw_boxes(cv::Mat mat_img, bbox_t* result_vec, std::vector<std::string> obj_names,int srcNum, int srcID=0,unsigned int wait_msec = 0) {
	for (int i = 0; i < srcNum; i++) {
		cv::Scalar color(60, 160, 260);
		cv::rectangle(mat_img, cv::Rect(result_vec[i].x, result_vec[i].y, result_vec[i].w, result_vec[i].h), color, 3);
		if (obj_names.size() > result_vec[i].obj_id)
			putText(mat_img, obj_names[result_vec[i].obj_id], cv::Point2f(result_vec[i].x, result_vec[i].y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
		//if (result_vec[i].track_id > 0)
			//putText(mat_img, std::to_string(result_vec[i].track_id), cv::Point2f(result_vec[i].x + 5, result_vec[i].y + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
	}
}
#endif	// OPENCV


void show_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names) {
	for (int i = 0; i < result_vec.size(); i++) {
		if (obj_names.size() > result_vec[i].obj_id) std::cout << obj_names[result_vec[i].obj_id] << " - ";
		std::cout << "obj_id = " << result_vec[i].obj_id << ",  x = " << result_vec[i].x << ", y = " << result_vec[i].y
			<< ", w = " << result_vec[i].w << ", h = " << result_vec[i].h
			<< std::setprecision(3) << ", prob = " << result_vec[i].score << std::endl;
	}
}

std::vector<std::string> objects_names_from_file(std::string const filename) {

	std::vector<std::string>dst;
	dst.clear();
	FILE *fp = fopen(filename.c_str(), "r");
	if (NULL == fp)
	{
		printf("failed to open dos.txt\n");
		return dst;
	}

	char szTest[256] = { 0 };
	int len = 0;
	while (!feof(fp))
	{
		memset(szTest, 0, sizeof(szTest));
		fgets(szTest, sizeof(szTest) - 1, fp); // °üº¬ÁË\n  
		std::string str = szTest;
		str = str.substr(0,str.size()-1);
		dst.push_back(str);
	}

	fclose(fp);

	return dst;
}

#define ZS_TIME


int main()
{
	std::string str = "/home/huluwa/darknet/";
	std::string cfg = str + "yolo.cfg";
	std::string weight = str + "backup/yolo_30000.weights";

	yDHANDLE srcYD0 = yDInit(const_cast<char*>(cfg.c_str()), const_cast<char*>(weight.c_str()),0);
	std::vector<std::string>  obj_names = objects_names_from_file(str + "voc.names");

	/*std::string l_str_Lable[38] = { "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
	"diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "truck", "tricar",
	"backmirror", "paperbox", "lightcover", "windglass", "hungs", "anusigns", "entrylisence", "safebelt", "plate", "carlight",
	"cartopwindow", "carrier", "newersign", "wheel", "layon"
	};*/

	std::string filename;
	filename = "/home/huluwa/darknet/5858.MOV";

	std::string const file_ext = filename.substr(filename.find_last_of(".") + 1);

	if (file_ext == "avi" || file_ext == "mp4" || file_ext == "mjpg" || file_ext == "MOV" || file_ext == "jpg")
	{
		IplImage* frameJPG = NULL;
		cv::Mat frame;
		if (file_ext == "jpg")
		{
			frameJPG = cvLoadImage(filename.c_str());			
			IplImage* frameResize = cvCreateImage(cvSize(608,608),8,3);
			cvResize(frameJPG, frameResize);
			int l_iNum = 0;
			bbox_t* result_vec0 = detect(srcYD0, frameResize, 0.4, 0.3, &l_iNum);
			draw_boxes(frameResize, result_vec0, obj_names, l_iNum, 0, 3);
			free(result_vec0);		
			cvSaveImage("/home/test.jpg", frameResize);
			cvReleaseImage(&frameJPG); frameJPG = NULL;
			cvReleaseImage(&frameResize); frameResize = NULL;
		}
		else
		{
 	            CvCapture *capture;
          	    capture = cvCreateFileCapture("/home/huluwa/darknet/5858.MOV");
		    IplImage* frameTemp=NULL; 
		    int l_iCounter = 0;
		    int nFrames = 0;
		// Create KCFTracker object
		KCFTracker tracker(1, 0, 1, 0);
		    while(1)
		    {
			frameTemp = cvQueryFrame(capture);
    			if(!frameTemp) break;

    			IplImage* frameResize = cvCreateImage(cvSize(608,608),8,3);
			cvResize(frameTemp, frameResize);

			int l_iNum = 0;
    			bbox_t* result_vec0 = detect(srcYD0, frameResize, 0.4, 0.3, &l_iNum);
    			draw_boxes(frameResize, result_vec0, obj_names, l_iNum,0, 3);
			
			if(nFrames == 0)
			{
				float xMin =   result_vec0[0].x;
				float yMin =   result_vec0[0].y;
				float width =  result_vec0[0].w;
				float height = result_vec0[0].h;
				cv::Mat frameMat = cv::Mat(frameTemp);
				tracker.init( cv::Rect(xMin, yMin, width, height), frameMat );
				rectangle( frameMat, cv::Point( xMin, yMin ), cv::Point( xMin+width, yMin+height), cv::Scalar( 0, 255, 255 ), 2, 8 );
			}

			else{
                                cv::Rect result;
				cv::Mat frameMat = cv::Mat(frameTemp);
				result = tracker.update(frameMat);
				rectangle( frameMat, cv::Point( result.x, result.y ), cv::Point( result.x+result.width,result.y+result.height), cv::Scalar( 0, 255, 255 ), 2, 8 );
			}

			nFrames++;
			free(result_vec0);
			//cvShowImage("test",frameResize);
			char l_imgPathName[256];
			sprintf(l_imgPathName,"/home/huluwa/test/t_%d.jpg",l_iCounter++);
			cvSaveImage(l_imgPathName, frameResize);
			if(cvWaitKey(33)>=0) break;
			cvReleaseImage(&frameResize); frameResize = NULL;
		    }
		}
	}
	
	return 0;
}




