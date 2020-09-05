#ifndef __YDT__  
#define __YDT__  

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

typedef long long yDHANDLE;

typedef struct bbox_t_{
	int x, y, w, h;
	int score;	
	unsigned int obj_id;
	char name[36];
}bbox_t;

typedef struct image_t_{
	int h;						// height
	int w;						// width
	int c;						// number of chanels (3 - for RGB)
	float *data;				// pointer to the image data
}image_t;

#ifdef __cplusplus
extern "C" {
#endif

yDHANDLE yDInit(const char* cfgPath, const char* weightPath, int gpuId);
bbox_t* detect(yDHANDLE srcHandle, IplImage* srcImg, float nms, float thresh, int* dstNum);
void yDunInit(yDHANDLE srcyDHandle);
int VeYOLO_MDIsBusy_(yDHANDLE h);
void VeYOLO_MDSetBusyState_(yDHANDLE h, int bState);
#ifdef __cplusplus
}
#endif
#endif  
