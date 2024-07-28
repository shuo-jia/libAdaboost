#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "image.h"

#ifdef IMG_JPEG
#include <jpeglib.h>
#include <setjmp.h>
#endif

/**
 * \file image.c
 * \brief 图像类型函数实现
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-27
 */
/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
#ifdef IMG_JPEG
/// 自定义 jpeglib 错误处理结构
struct myjpeg_error_mgr {
	struct jpeg_error_mgr err_part;		// 用于 jpeglib 错误处理
	jmp_buf setjmp_part;			// 用于 setjmp 返回
};
#endif

/*******************************************************************************
 * 				   宏函数定义
 ******************************************************************************/
/// 计算模板内的像素平均值
#define RECT_AVG(val, img_ptr, d, i, j, size)					\
	do {									\
		val = 0;							\
		for (int start_y = i; start_y < i + size; ++start_y)		\
			for (int start_x = j; start_x < j + size; ++start_x)	\
				val += img_ptr[start_y][start_x][d];		\
		val = val / (size * size) + 0.5;				\
	} while(0)

/*******************************************************************************
 * 				  静态函数原型
 ******************************************************************************/
#ifdef IMG_JPEG
/// libjpeg 库的错误处理函数
static void myjpeg_error_exit (j_common_ptr cinfo);
#endif

/// 将图像 img 的一个矩形区域（rect）填充到 sub_img 中
static void fill_sub_image (unsigned char *sub_img, const struct image *img,
		const struct rectangle *rect);

/// 读取 pgm 文件头部
static bool read_pgm_head (FILE *file, char *format, int *width, int *height);

/// 跳过 pgm 文件注释行（一行），如果该行并非注释，则返回 false
static bool read_pgm_skip (FILE *file);

// 读取 pgm 文件 ASCII 形式的图像
static bool read_pgm_ascii (FILE *file, struct image *img, int width, int height);

/*******************************************************************************
 * 				    函数实现
 ******************************************************************************/
struct image * new_image (int height, int width, int depth)
{
	struct image *img = malloc(sizeof(struct image) +
			sizeof(unsigned char) * height * width * depth);
	img->height = height;
	img->width = width;
	img->depth = depth;
	return img;
}

#ifdef IMG_JPEG
struct image * imread_jpeg (const char * name)
{
	FILE * file = fopen(name, "rb");
	struct image *img = NULL;
	if (file == NULL)
		return NULL;
	struct jpeg_decompress_struct cinfo;
	struct myjpeg_error_mgr jerr;
	// 设置错误处理函数
	cinfo.err = jpeg_std_error(&jerr.err_part);
	jerr.err_part.error_exit = myjpeg_error_exit;

	// 后续如若 libjpeg 出现错误，则将返回此处
	if (setjmp(jerr.setjmp_part)) {
		jpeg_destroy_decompress(&cinfo);
		free(img);
		fclose(file);
		return NULL;
	}

	jpeg_create_decompress(&cinfo);		// 初始化
	jpeg_stdio_src(&cinfo, file);		// 输入文件
	jpeg_read_header(&cinfo, TRUE);		// 读取一些参数
	jpeg_start_decompress(&cinfo);		// 初始化解压缩循环的状态以及申请内存
	// 每行的元素数量（R,G,B,R,G,B,...排列）
	int row_stride = cinfo.output_width * cinfo.output_components;
	int size = cinfo.output_height * row_stride;
	// 用于保存彩色图，大小为 高度*宽度*深度
	img = malloc(sizeof(struct image) + sizeof(unsigned char) * size);
	if (img == NULL) {
		jpeg_finish_decompress (&cinfo);
		jpeg_destroy_decompress (&cinfo);
		fclose (file);
		return NULL;
	}
	// 获取解压数据
	JSAMPARRAY buffer;			// 用于暂存数据
	buffer = (*cinfo.mem->alloc_sarray)
		((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);
	img->height = cinfo.output_height;
	img->width = cinfo.output_width;
	img->depth = cinfo.output_components;
	// 循环扫描每一行
	int i = 0;
	IMAGE_PTR_TYPE(img_ptr, img) = IMAGE_PTR(img);
	while (cinfo.output_scanline < cinfo.output_height) {
		jpeg_read_scanlines(&cinfo, buffer, 1);
		memcpy (img_ptr[i], buffer[0], sizeof(unsigned char) * row_stride);
		i++;
	}

	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	fclose(file);
	return img;
}
#endif

// 读取图像文件（pgm 格式）
struct image * imread_pgm (const char * name)
{
	struct image * img = NULL;
	char format[3];
	int width, height;
	FILE * file = fopen (name, "rb");
	if (file == NULL)
		return NULL;

	// 读取文件头部
	if (read_pgm_head (file, format, &width, &height) == false)
		goto read_head_err;

	unsigned int len = width * height;
	img = malloc (sizeof(struct image) + sizeof(unsigned char) * len);
	if (img == NULL)
		goto read_head_err;

	// 图像以二进制形式保存的情形
	if (!strcmp(format, "P5")) {
		if (fread (img->img, sizeof(unsigned char), len, file) < len)
			goto read_data_err;
	}
	// 图像以 ASCII 码形式保存的情形
	else if (!strcmp(format, "P2")) {
		if (read_pgm_ascii (file, img, width, height) == false)
			goto read_data_err;
	}
	else
		goto read_data_err;

	img->width = width;
	img->height = height;
	img->depth = 1;
	fclose (file);
	return img;

read_data_err:
	free(img);
read_head_err:
	fclose(file);
	return NULL;
}

struct image * get_sub_image (const struct image *img, const struct rectangle *rect)
{
	unsigned long long size = rect->height * rect->width * img->depth;
	struct image *sub_img = malloc(sizeof(struct image) +
				       sizeof(unsigned char) * size);
	if (sub_img == NULL)
		return NULL;
	fill_sub_image (sub_img->img, img, rect);
	sub_img->height = rect->height;
	sub_img->width = rect->width;
	sub_img->depth = img->depth;
	return sub_img;
}

void cropping_image (struct image *img, const struct rectangle *rect)
{
	int i, j, k;
	unsigned char *target = img->img;
	IMAGE_PTR_TYPE(img_ptr, img) = IMAGE_PTR(img);
	for (i = rect->start_y; i < rect->start_y + rect->height; ++i)
		for (j = rect->start_x; j < rect->start_x + rect->width; ++j)
			for (k = 0; k < img->depth; ++k)
				*target++ = img_ptr[i][j][k];

	img->height = rect->height;
	img->width = rect->width;
}

void reduced_image (struct image *img, int height, int width)
{
	float h_rate = (float)img->height / height;
	float w_rate = (float)img->width / width;
	IMAGE_PTR_TYPE(img_ptr, img) = IMAGE_PTR(img);
	unsigned char *ptr = img->img;

	int i, j, k;
	float posi_i = 0;
	float posi_j = 0;
	for (i = 0; i < height; ++i) {
		posi_j = 0;
		for (j = 0; j < width; ++j) {
			for (k = 0; k < img->depth; ++k)
				*ptr++ = img_ptr[(int)posi_i][(int)posi_j][k];
			posi_j += w_rate;
		}
		posi_i += h_rate;
	}
	img->height = height;
	img->width = width;
}

void make_grey_image (struct image *img)
{
	unsigned int total;
	unsigned char *tgt_ptr = img->img;
	IMAGE_PTR_TYPE(img_ptr, img) = IMAGE_PTR(img);
	int i, j, k;
	for (i = 0; i < img->height; ++i)
		for (j = 0; j < img->width; ++j) {
			total = 0;
			for (k = 0; k < img->depth; ++k)
				total += img_ptr[i][j][k];
			*tgt_ptr = total / img->depth;
			++tgt_ptr;
		}
	img->depth = 1;
}

struct image * fit_image (struct image *img)
{
	struct image *new_ptr = realloc (img, sizeof(struct image) +
		sizeof(unsigned char) * img->height * img->width * img->depth);
	if (new_ptr == NULL)
		return img;
	return new_ptr;
}

struct image * mean_filter (const struct image *img, int size)
{
	struct image * result = new_image (img->height, img->width, img->depth);
	if (result == NULL)
		return NULL;

	int i, j, d;
	unsigned int total;
	int left_half = size / 2;			// 像素点左（上）侧模板长度
	int right_half = size - size / 2 - 1;		// 像素点右（下）侧模板长度
	IMAGE_PTR_TYPE(img_ptr, img) = IMAGE_PTR(img);
	IMAGE_PTR_TYPE(rst_ptr, result) = IMAGE_PTR(result);

	// 图像上下边缘填充为原图像
	memcpy (rst_ptr, img_ptr, sizeof(unsigned char) * img->width * img->depth
				  * left_half);
	memcpy (rst_ptr[img->height - right_half], img_ptr[img->height - right_half],
		sizeof(unsigned char) * img->width * img->depth * right_half);
	for (i = left_half; i < img->height - right_half; ++i) {
		// 均值滤波
		for (j = left_half; j < img->width - right_half; ++j)
			for (d = 0; d < img->depth; ++d) {
				RECT_AVG(total, img_ptr, d, i - left_half,
					j - left_half, size);
				rst_ptr[i][j][d] = total;
			}
		// 填充左右边缘
		for (j = 0; j < left_half; ++j)
			for (d = 0; d < img->depth; ++d)
				rst_ptr[i][j][d] = img_ptr[i][j][d];
		for (j = img->width - right_half; j < img->width; ++j)
			for (d = 0; d < img->depth; ++d)
				rst_ptr[i][j][d] = img_ptr[i][j][d];
	}
	return result;
}

/*******************************************************************************
 * 				  静态函数实现
 ******************************************************************************/
#ifdef IMG_JPEG
void myjpeg_error_exit (j_common_ptr cinfo)
{
	char msg[JMSG_LENGTH_MAX];
	(*cinfo->err->format_message)(cinfo, msg);
	fprintf(stderr, "Error: %s\n", msg);

	struct myjpeg_error_mgr * perr = (struct myjpeg_error_mgr *) cinfo->err;
	longjmp(perr->setjmp_part, 1);
}
#endif

void fill_sub_image (unsigned char *sub_img, const struct image *img,
		const struct rectangle *rect)
{
	const IMAGE_PTR_TYPE(img_ptr, img) = IMAGE_PTR(img);
	int len = rect->width * img->depth;
	for (int i = rect->start_y; i < rect->start_y + rect->height; ++i) {
		memcpy (sub_img, img_ptr[i][rect->start_x],
			sizeof(unsigned char) * len);
		sub_img += len;
	}
}

bool read_pgm_head (FILE *file, char *format, int *width, int *height)
{
	int max_val;
	if (fscanf(file, "%2s", format) != 1)
		return false;
	while (fscanf (file, "%d", width) != 1)
		if (! read_pgm_skip (file))
			return false;
	while (fscanf (file, "%d", height) != 1)
		if (! read_pgm_skip (file))
			return false;
	while (fscanf (file, "%d", &max_val) != 1)
		if (! read_pgm_skip (file))
			return false;

	// 去除行尾空白字符，避免影响二进制数据读取
	while (fgetc(file) != '\n')
		continue;
	return true;
}

bool read_pgm_skip (FILE *file)
{
	char ch;
	if (fscanf(file, " %c", &ch) == 1 && ch == '#')
		while (fgetc(file) != '\n')
			continue;
	else
		return false;

	return true;
}

bool read_pgm_ascii (FILE *file, struct image *img, int width, int height)
{
	int i, j;
	unsigned int val;
	unsigned char *ptr = img->img;
	for (i = 0; i < height; ++i)
		for (j = 0; j < width; ++j) {
			if (fscanf (file, "%u", &val) < 1)
				return false;
			*ptr++ = val;
		}
	return true;
}
