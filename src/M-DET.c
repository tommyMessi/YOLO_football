#include "M-DET.h"

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"



#include "network.h"

#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"
#include "blas.h"


typedef struct{
	float **probs;
	box * boxes;
	network net;
	int m_bBusyState;
}detector_gpu_t;

#define ZSMAX(a,b) ((a) > (b) ? (a) : (b)) 



void zs_check_error(cudaError_t status)
{
    //cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    }
    if (status2 != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    }
}

void zs_cuda_set_device(int n)
{
    gpu_index = n;
    printf("zs cuda fenpei = %d \n",n);
    cudaError_t status = cudaSetDevice(n);
    zs_check_error(status);
} 

yDHANDLE yDInit(const char* cfgPath, const char* weightPath, int gpuId)
{
	zs_cuda_set_device(gpuId);
	detector_gpu_t * dstDetector = (detector_gpu_t *)calloc(1, sizeof(detector_gpu_t));//new detector_gpu_t;
	
    dstDetector->net.gpu_index = gpuId;

	dstDetector->net = parse_network_cfg(cfgPath);

    if (weightPath)
	{
		load_weights(&dstDetector->net, weightPath);
	}
	set_batch_network(&dstDetector->net, 1);

	layer l = dstDetector->net.layers[dstDetector->net.n - 1];

	dstDetector->boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));//new box[l.w*l.h*l.n];
	dstDetector->probs = (float **)calloc(l.h*l.w*l.c, sizeof(float *));;
	int j = 0;
        for ( j = 0; j < l.w*l.h*l.n; ++j) 
		dstDetector->probs[j] = (float *)calloc(l.classes + 1, sizeof(float));//new float[l.classes + 1];

	dstDetector->net.gpu_index = gpuId;

	return (yDHANDLE)(dstDetector);
}

static image_t make_empty_image_(int w, int h, int c)
{
	image_t out;
	out.data = 0;
	out.h = h;
	out.w = w;
	out.c = c;
	return out;
}

static image_t make_image_custom(int w, int h, int c)
{
	image_t out = make_empty_image_(w, h, c);
	//out.data = new float[h*w*c];
	out.data = (float *)calloc(h*w*c, sizeof(float));
        return out;
}

static image_t ipl_to_image_t(IplImage* src)
{
	unsigned char *data = (unsigned char *)src->imageData;
	int h = src->height;
	int w = src->width;
	int c = src->nChannels;
	int step = src->widthStep;
	image_t out = make_image_custom(w, h, c);
	int i, j, k, count = 0;;

	for (k = 0; k < c; ++k) {
		for (i = 0; i < h; ++i) {
			for (j = 0; j < w; ++j) {
				out.data[count++] = data[i*step + j*c + k] / 255.;
			}
		}
	}
	return out;
}

static void rgbgr_image_t(image_t im)
{
	int i;
	for (i = 0; i < im.w*im.h; ++i) {
		float swap = im.data[i];
		im.data[i] = im.data[i + im.w*im.h * 2];
		im.data[i + im.w*im.h * 2] = swap;
	}
}

static image_t mat_to_image(IplImage* img)
{
	image_t im_small = ipl_to_image_t(img);
	rgbgr_image_t(im_small);
	return im_small;
}

bbox_t* detect(yDHANDLE srcHandle, IplImage* srcImg, float nms, float thresh, int* dstNum)
{
	detector_gpu_t * srcDetector = (detector_gpu_t*)srcHandle;
	
	if (!srcDetector) return NULL;

	char l_str_Lable[38][36] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
		"diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "truck", "tricar",
		"backmirror", "paperbox", "lightcover", "windglass", "hungs", "anusigns", "entrylisence", "safebelt", "plate", "carlight",
		"cartopwindow", "carrier", "newersign", "wheel", "layon","background"};

	clock_t start;
	clock_t end;
	start = clock();

	IplImage* frameResize = cvCreateImage(cvSize(608, 608), 8, 3);
	cvResize(srcImg, frameResize, 1);
	/*int l_iMax = (srcImg->width > srcImg->height) ? srcImg->width : srcImg->height;

	IplImage* pImage = cvCreateImage(cvSize(l_iMax, l_iMax), srcImg->depth, srcImg->nChannels);
	cvZero(pImage);
	CvRect roi = cvRect(0, 0, srcImg->width, srcImg->height);
	cvSetImageROI(srcImg, roi);
	cvSetImageROI(pImage, roi);
	cvCopy(srcImg, pImage,NULL);
	cvResetImageROI(srcImg);
	cvResetImageROI(pImage);
	cvResize(pImage, frameResize,1);*/

	image_t img = mat_to_image(frameResize);

	cvReleaseImage(&frameResize); frameResize = NULL;
	//cvReleaseImage(&pImage); pImage = NULL;

	image im;
	im.c = img.c;
	im.data = img.data;
	im.h = img.h;
	im.w = img.w;
	
	end = clock();
	printf("convet image time is : %f ms\n", sec(end - start));
	
	start = clock();
	//cudaSetDevice(srcDetector->net.gpu_index);
	zs_cuda_set_device(srcDetector->net.gpu_index);
        printf("use for test. \n");
        float *X = im.data;
	network_predict(srcDetector->net, X);
	end = clock();
	printf("prefill time is : %f ms\n", sec(end - start));

	start = clock();
	layer l = srcDetector->net.layers[srcDetector->net.n - 1];
    get_region_boxes(l, im.w, im.h, srcDetector->net.w, srcDetector->net.h, thresh, srcDetector->probs, srcDetector->boxes, NULL, 0, 0, 0.5, 1);
	//get_region_boxes(l, im.w, im.h, srcDetector->net.w, srcDetector->net.h, thresh, srcDetector->probs, srcDetector->boxes, 0, 0, 0.5, 1);
	end = clock();
	printf("prefill time 2 is : %f ms\n", sec(end - start));
	start = clock();
	//get_region_boxes_J(l, 1, 1, thresh, srcDetector->probs, srcDetector->boxes, 0, 0);
	if (nms) do_nms_obj(srcDetector->boxes, srcDetector->probs, l.w*l.h*l.n, l.classes, nms);
	end = clock();
	printf("prefill time 3 is : %f ms\n", sec(end - start));
	

    bbox_t* bbox_vec = (bbox_t *)calloc(128, sizeof(bbox_t));

    (*dstNum) = 0;
    int i = 0;
	for ( i = 0; i < (l.w*l.h*l.n); ++i) 
	{
		box b = srcDetector->boxes[i];
		int const obj_id = max_index(srcDetector->probs[i], l.classes);
		float const prob = srcDetector->probs[i][obj_id];

		if (prob > thresh)
		{
			bbox_t bbox;
			bbox_vec[(*dstNum)].x = ZSMAX((double)0, (b.x - b.w / 2.)*srcImg->width/*im.w*/);
			bbox_vec[(*dstNum)].y = ZSMAX((double)0, (b.y - b.h / 2.)*srcImg->height/*im.h*/);
			bbox_vec[(*dstNum)].w = b.w*srcImg->width/*im.w*/;
			bbox_vec[(*dstNum)].h = b.h*srcImg->height/*im.h*/;
			bbox_vec[(*dstNum)].obj_id = obj_id;
			bbox_vec[(*dstNum)].score = prob*100;
			strcpy(bbox_vec[(*dstNum)].name, l_str_Lable[obj_id]);
			(*dstNum)++;
		}
	}

	if (img.data) 
	{
		//delete[] img.data;
                free(img.data);
		img.c = 0; img.h = 0; img.w = 0;
	}
	
	return bbox_vec;
}

void yDunInit(yDHANDLE srcyDHandle)
{
	detector_gpu_t * detector_gpu = (detector_gpu_t*)srcyDHandle;
	layer l = detector_gpu->net.layers[detector_gpu->net.n - 1];
        int j = 0;
	for (j = 0; j < l.w*l.h*l.n; ++j)
	{
		if (detector_gpu->probs[j])
		{
			free(detector_gpu->probs[j]);
			//delete[] detector_gpu->probs[j];
			//detector_gpu->probs[j] = NULL;
		}
	}
	if (detector_gpu->probs)
	{
		free(detector_gpu->probs);
		//delete[]detector_gpu->probs;
		//detector_gpu->probs = NULL;
	}
	
	if (detector_gpu->boxes)
            free(detector_gpu->boxes);	    
//delete [] detector_gpu->boxes;

	int old_gpu_index;
#ifdef GPU
	cudaGetDevice(&old_gpu_index);
	cudaSetDevice(detector_gpu->net.gpu_index);
#endif

	free_network(detector_gpu->net);

#ifdef GPU
	cudaSetDevice(old_gpu_index);
#endif
	//delete[] detector_gpu;
	free(detector_gpu);
}

int VeYOLO_MDIsBusy_(yDHANDLE h)
{
	detector_gpu_t * detector_gpu = (detector_gpu_t*)h;
	if (detector_gpu)
	{
		return detector_gpu->m_bBusyState;
	}
	return 0;
}

void VeYOLO_MDSetBusyState_(yDHANDLE h, int bState)
{
	detector_gpu_t * detector_gpu = (detector_gpu_t*)h;
	if (detector_gpu)
	{
		detector_gpu->m_bBusyState = bState;
	}
}
