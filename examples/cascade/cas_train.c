#include <stdlib.h>
#include <string.h>
#include "image.h"
#include "cascade.h"

/*******************************************************************************
 *				   宏常量定义
 ******************************************************************************/
// 文件名最大长度
#define MAX_FILENAME 128
// 图片所在目录
#define DIRPATH "./BioID-FaceDatabase-V1.2/"
// 标注文件
#define MARKPATH "./p_mark"
// 每个 AdaBoost 分类器的检测率、假阳率
#define DET_RATE 0.99
#define FP_RATE 0.6
// 最大假阳率
#define MAX_FP_RATE 1E-8
// 训练集占比（其余设置为验证集）
#define TRAIN_SET_PERCENT 0.7
// 训练图像尺寸（像素）
#define FACE_SIZE 24
// 阳性样本数量
#define P_NUM 1000
// 阴性样本数量
#define N_NUM 100000
// 模型保存路径
#define MODEL_PATH "./cascade_data"

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
struct sp_args {
	FILE * mark;			// 标注文件
	fpos_t cursor_face;		// (最后一次获取人脸样本后)标注文件指针
	struct image *img;		// 当前包含人脸的图片
	int face_id;			// 当前人脸图片的 id

	struct image *non_faces[4];	// 从一张（包含人脸）图片提取的 4 张非人
					// 脸图片
	fpos_t cursor_non_face;		// (最后一次获取非人脸样本后)标注文件指针
	int posi;			// non_faces 数组当前索引
	int img_id;			// 当前使用的人脸图片 id（用于提取非人脸
					// 样本）
	int non_face_id;		// 当前非人脸样本 id
};

/*******************************************************************************
 *				  静态函数声明
 ******************************************************************************/
// 初始化回调函数参数
static bool init_args (struct sp_args *args, const char * mark_file);

// 释放回调函数参数
static void free_args (struct sp_args *args);

// 回调函数：获取人脸图片
static const unsigned char *get_face(imgsz_t * h, imgsz_t * w,
				     struct cas_rect *rect, void *args);
// 回调函数：获取非人脸图片
static const unsigned char *get_non_face(imgsz_t * h, imgsz_t * w, num_t * id,
					 void *args);
// \brief 从标注文件获取图像
// \param[in] file      已打开的标注文件
// \param[in, out] posi 上次读取后文件指针所处的位置，函数返回后被置为当前位置
// \param[out] rect     用于保存人脸矩形框
// \return 返回读取的图像；如果失败则返回 NULL
static struct image *get_marked_img(FILE * file, fpos_t * posi,
				    struct cas_rect *rect);

/*******************************************************************************
 *				    函数实现
 ******************************************************************************/
int main(void)
{
	struct sp_args args;
	struct cascade cascade;
	struct haar_ada_handles handles;
	if (!init_args (&args, MARKPATH))
		exit(EXIT_FAILURE);

	ada_set_haar (&handles, ADA_ASYM_IMP, ADA_GA);
	if (!cas_train (&cascade, DET_RATE, FP_RATE, MAX_FP_RATE,
			TRAIN_SET_PERCENT, P_NUM, N_NUM, FACE_SIZE, &args,
			get_face, get_non_face, &handles)) {
		fprintf(stderr, "Training Error.\n");
		goto train_err;
	}

	FILE * file = fopen (MODEL_PATH, "wb");
	cas_write (&cascade, file, &handles);
	fclose (file);
	cas_free (&cascade, &handles);
	free_args (&args);
	return 0;

train_err:
	free_args (&args);
	exit(EXIT_FAILURE);
}

/*******************************************************************************
 *				  静态函数实现
 ******************************************************************************/
bool init_args(struct sp_args *args, const char *mark_file)
{
	args->img = NULL;
	memset(args, 0, sizeof(struct sp_args));
	if ((args->mark = fopen(MARKPATH, "r")) == NULL)
		return false;
	if (fgetpos(args->mark, &args->cursor_face)
	    || fgetpos(args->mark, &args->cursor_non_face)) {
		fclose(args->mark);
		return false;
	}
	args->posi = 4;
	return true;
}

// 释放回调函数参数
void free_args (struct sp_args *args)
{
	fclose (args->mark);
	free (args->img);
	for (int i = 0; i < 4; ++i)
		free (args->non_faces[i]);
}

const unsigned char *get_face(imgsz_t * h, imgsz_t * w, struct cas_rect *rect,
			      void *v_args)
{
	struct sp_args *args = v_args;
	struct image * img;
	if (args->face_id >= P_NUM) {
		args->face_id = 0;
		rewind (args->mark);
		fgetpos (args->mark, &args->cursor_face);
	}
	if (!(img = get_marked_img(args->mark, &args->cursor_face, rect)))
		return NULL;
	*h = img->height;
	*w = img->width;
	free (args->img);
	args->img = img;
	++args->face_id;
	return img->img;
}

const unsigned char *get_non_face(imgsz_t * h, imgsz_t * w, num_t * id,
				  void *v_args)
{
	struct sp_args *args = v_args;
	struct image * img;
	struct cas_rect face;
	struct rectangle rect;
	if (args->posi > 3 || args->non_faces[args->posi] == NULL) {
		if (args->img_id >= P_NUM) {
			args->img_id = 0;
			args->non_face_id = 0;
			rewind (args->mark);
			fgetpos (args->mark, &args->cursor_non_face);
		}
		if (!(img = get_marked_img(args->mark, &args->cursor_non_face, &face)))
			return NULL;
		for (int i = 0; i < 4; ++i) {
			free (args->non_faces[i]);
			args->non_faces[i] = NULL;
		}
		int index = 0;
		// 以人脸中线（竖线）为分界线，将图片分为左、右两张非人脸图片
		rect.start_x = 0;
		rect.start_y = 0;
		rect.height = img->height;
		if ((rect.width = face.start_x + face.len / 2.0) >= FACE_SIZE)
			if (!(args->non_faces[index++] = get_sub_image (img, &rect)))
				goto err;
		rect.start_x += rect.width;
		if ((rect.width = img->width - rect.start_x) >= FACE_SIZE)
			if (!(args->non_faces[index++] = get_sub_image (img, &rect)))
				goto err;
		// 以人脸中线（水平线）为分界线，将图片分为上、下两张非人脸图片
		rect.start_x = rect.start_y = 0;
		rect.width = img->width;
		if ((rect.height = face.start_y + face.len / 2.0) >= FACE_SIZE)
			if (!(args->non_faces[index++] = get_sub_image (img, &rect)))
				goto err;
		rect.start_y = rect.height;
		if ((rect.height = img->height - rect.start_y) >= FACE_SIZE)
			if (!(args->non_faces[index++] = get_sub_image (img, &rect)))
				goto err;
		++args->img_id;
		args->posi = 0;
		free (img);
	}
	++args->non_face_id;
	*h = args->non_faces[args->posi]->height;
	*w = args->non_faces[args->posi]->width;
	*id = args->non_face_id;
	return args->non_faces[args->posi++]->img;
err:
	free (img);
	return NULL;
}

struct image *get_marked_img(FILE * file, fpos_t * posi, struct cas_rect *rect)
{
	struct rectangle face;
	struct image *img = NULL;
	char filename[MAX_FILENAME] = DIRPATH;
	char *f_ptr = filename + strlen(filename);
	fsetpos (file, posi);
	if (fscanf(file, "%s %d %d %d %d", f_ptr, &face.start_x, &face.start_y,
	     &face.height, &face.width) < 5)
		return NULL;
	fgetpos (file, posi);
	if ((img = imread_pgm(filename)) == NULL)
		return NULL;
	rect->start_x = face.start_x;
	rect->start_y = face.start_y;
	rect->len = face.height;
	return img;
}
