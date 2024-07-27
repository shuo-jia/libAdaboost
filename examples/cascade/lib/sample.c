#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "image.h"
#include "sample.h"
#include "cascade.h"

/*******************************************************************************
 * 				  静态函数声明
 ******************************************************************************/
// 为 struct sample 结构体成员申请内存空间并初始化
static bool alloc_sample (struct sample * sample, num_t m, imgsz_t face_size);

// 为 struct sample 结构体成员缩小空间，原有数据被保留，超出范围的数据被截断
static bool reduce_sample (struct sample * sample, num_t m);

// 将图片保存到（二维）数组中
static void img2arr (sample_t arr[], struct image * img);

// 将图像 img 中矩形框（rect）框内部分转换为样本集 sp 的第 i 个积分图
static inline bool img2sample (struct sample *sp, const struct image *img,
		const struct rectangle *rect, imgsz_t face_size, num_t i);

// 从图片背景中随机生成若干阴性（非人脸）样本
// sp: 样本集
// posi: 指示当前样本序号，即从 sp->X[posi] 开始保存阴性样本（X2、Y 同理）
// n: 所需生成的样本数量
// face_size: 人脸尺寸（像素），阴性样本保持相同尺寸
// img: 保存图片的结构体
// face: 指示图片中人脸位置、大小的矩形框
// fname: img 图片的文件名
// mark: 用于保存生成的矩形框及文件名，为 NULL 时表示不保存
// 返回值：成功则返回真，否则返回假
static bool get_rand_neg (struct sample *sp, num_t posi, num_t n,
		imgsz_t face_size, const struct image *img,
		const struct rectangle *face, const char *fname, FILE * mark);

// 统计标注文件矩形框个数
static num_t mark_count (FILE * mark);

// 从标注文件中读取样本保存到 sp 的第 posi 个样本及其后，label 指示样本标签
// 返回成功读取的样本数量
static num_t fill_sample (struct sample * sample, num_t posi, num_t max_count,
		imgsz_t face_size, label_t label, FILE * mark,
		const char * dir_path);
/*******************************************************************************
 * 				    函数定义
 ******************************************************************************/
bool get_rand_sample (struct sample * sample, num_t max_posi, num_t neg_per_posi, 
		imgsz_t face_size, FILE * p_mark, FILE * n_mark,
		const char * dir_path)
{
	char filename[MAX_FILENAME];
	struct rectangle rect;
	struct image *img = NULL;

	num_t count = mark_count (p_mark);
	if (count < max_posi)
		max_posi = count;
	if (!alloc_sample (sample, max_posi * (neg_per_posi + 1), face_size))
		return false;
	strcpy(filename, dir_path);
	char *f_ptr = filename + strlen(filename);
	num_t n, i;
	for (n = 0, i = 0; n < max_posi; ++n) {
		if (fscanf (p_mark, "%s %d %d %d %d", f_ptr, &rect.start_x,
			&rect.start_y, &rect.height, &rect.width) < 5)
			goto read_err;
		if ((img = imread_pgm (filename)) == NULL)
			goto read_err;
		// 获取 neg_per_posi 个非人脸样本
		if (!get_rand_neg (sample, i, neg_per_posi, face_size, img,
					&rect, f_ptr, n_mark))
			goto gen_sp_err;
		i += neg_per_posi;
		if (!img2sample (sample, img, &rect, face_size, i))
			goto gen_sp_err;
		sample->Y[i] = 1;
		++i;
		free (img);
	}

	return true;
gen_sp_err:
	free (img);
read_err:
	free_sample (sample);
	return false;
}

bool get_mark_sample (struct sample * sample, num_t max_posi, imgsz_t face_size,
		FILE * p_mark, FILE * n_mark, const char * dir_path)
{
	// 统计图像文件个数
	num_t p_count = mark_count (p_mark);
	if (p_count > max_posi)
		p_count = max_posi;
	num_t n_count = mark_count (n_mark);
	if (!alloc_sample (sample, p_count + n_count, face_size))
		return false;

	if (fill_sample (sample, 0, p_count, face_size, 1, p_mark, dir_path) < p_count)
		goto err;
	if (fill_sample (sample, p_count, n_count, face_size, -1, n_mark, dir_path) < n_count)
		goto err;
	return true;
err:
	free_sample (sample);
	return false;
}

void free_sample (struct sample * sample)
{
	for (num_t i = 0; i < sample->m; ++i) {
		free (sample->X[i]);
		free (sample->X2[i]);
	}
	free (sample->X);
	free (sample->X2);
	free (sample->Y);
}

void rect_cast (struct rectangle *rect, const struct cas_rect * cas_rect)
{
	rect->start_x = cas_rect->start_x;
	rect->start_y = cas_rect->start_y;
	rect->height = cas_rect->height;
	rect->width = cas_rect->width;
}

void rect_cast_r (struct cas_rect * cas_rect, const struct rectangle *rect)
{
	cas_rect->start_x = rect->start_x;
	cas_rect->start_y = rect->start_y;
	cas_rect->height = rect->height;
	cas_rect->width = rect->width;
}

/*******************************************************************************
 * 				  静态函数定义
 ******************************************************************************/
bool alloc_sample (struct sample * sample, num_t m, imgsz_t face_size)
{
	sample->X = malloc (sizeof(sample_t *) * m);
	sample->X2 = malloc (sizeof(sample_t *) * m);
	sample->Y = malloc (sizeof(label_t) * m);
	if (!sample->X || !sample->X2 || !sample->Y)
		goto malloc_err;
	num_t n;
	imgsz_t len = face_size * face_size;
	for (n = 0; n < m; ++n) {
		if (!(sample->X[n] = (sample_t *)malloc(sizeof(sample_t)*len)))
			goto malloc_arrs_err;
		if (!(sample->X2[n] = (sample_t *)malloc(sizeof(sample_t)*len))){
			free (sample->X[n]);
			goto malloc_arrs_err;
		}
	}
	sample->m = m;
	return true;

malloc_arrs_err:
	for (num_t i = 0; i < n; ++i) {
		free (sample->X[i]);
		free (sample->X2[i]);
	}
malloc_err:
	free (sample->X);
	free (sample->X2);
	free (sample->Y);
	return false;
}

bool reduce_sample (struct sample * sample, num_t m)
{
	if (m > sample->m)
		return false;
	if (m <= sample->m)
		for (num_t i = m; i < sample->m; ++i) {
			free (sample->X[i]);
			free (sample->X2[i]);
		}
	sample->m = m;

	void * ptr;
	if (!(ptr = realloc (sample->X, sizeof(sample_t *) * m)))
		return false;
	sample->X = (sample_t **) ptr;
	if (!(ptr = realloc (sample->X2, sizeof(sample_t *) * m)))
		return false;
	sample->X2 = (sample_t **) ptr;
	if (!(ptr = realloc (sample->Y, sizeof(label_t) * m)))
		return false;
	sample-> Y = (label_t *)ptr;

	return false;
}

void img2arr (sample_t arr[], struct image * img)
{
	IMAGE_PTR_TYPE (ptr, img) = IMAGE_PTR(img);
	imgsz_t index = 0;
	for (imgsz_t i = 0; i < img->height; ++i)
		for (imgsz_t j = 0; j < img->width; ++j)
			arr[index++] = ptr[i][j][0];
}

bool img2sample (struct sample *sp, const struct image *img,
		const struct rectangle *rect, imgsz_t face_size, num_t i)
{
	struct image * subimg = NULL;
	// 裁剪图像，选取矩形框部分
	if ((subimg = get_sub_image (img, rect)) == NULL)
		return false;
	// 缩小图像至人脸大小
	reduced_image (subimg, face_size, face_size);
	// 复制图像至数组
	img2arr (sp->X[i], subimg);
	img2arr (sp->X2[i], subimg);
	// 转换为积分图
	cas_intgraph (face_size, face_size, (void *)sp->X[i]);
	cas_intgraph2 (face_size, face_size, (void *)sp->X2[i]);

	free (subimg);
	return true;
}

bool get_rand_neg (struct sample *sp, num_t posi, num_t n, imgsz_t face_size,
		const struct image *img, const struct rectangle *face,
		const char *fname, FILE * mark)
{
	struct rectangle rand_rect;
	struct cas_rect rect1, rect2;

	rect_cast_r (&rect1, face);
	imgsz_t min_size = (img->height > img->width) ? img->width : img->height;
	for (num_t i = 0; i < n; ++i) {
		do {
			rect2.width = rand() % (min_size - face_size)
					+ face_size;
			rect2.height = rect2.width;
			rect2.start_x = rand() % (img->width -
						      rect2.width);
			rect2.start_y = rand() % (img->height -
						      rect2.height);
		} while (IoU (&rect1, &rect2) > 0.3);
		rect_cast (&rand_rect, &rect2);
		if (!img2sample (sp, img, &rand_rect, face_size, posi))
			return false;
		if (mark != NULL && fprintf (mark, "%s %d %d %d %d\n", fname,
					rect2.start_x, rect2.start_y,
					rect2.height, rect2.width) < 5)
			return false;
		sp->Y[posi] = -1;
		++posi;
	}
	return true;
}

num_t mark_count (FILE * mark)
{
	fpos_t pos;
	fgetpos (mark, &pos);
	num_t count = 0;
	char filename [MAX_FILENAME];
	struct rectangle rect;
	while (fscanf (mark, "%s %d %d %d %d", filename, &rect.start_x,
				&rect.start_y, &rect.height, &rect.width) == 5)
		++count;
	fsetpos (mark, &pos);
	return count;
}

num_t fill_sample (struct sample * sample, num_t posi, num_t max_count,
		imgsz_t face_size, label_t label, FILE * mark,
		const char * dir_path)
{
	char filename[MAX_FILENAME];
	struct rectangle rect;
	struct image *img = NULL;

	strcpy(filename, dir_path);
	char *f_ptr = filename + strlen(filename);
	num_t i;
	for (i = 0; i < max_count; ++i) {
		if (fscanf (mark, "%s %d %d %d %d", f_ptr, &rect.start_x,
			&rect.start_y, &rect.height, &rect.width) < 5)
			return i;
		if ((img = imread_pgm (filename)) == NULL)
			return i;
		if (!img2sample (sample, img, &rect, face_size, posi)) {
			free (img);
			return i;
		}
		sample->Y[posi] = label;
		++posi;

		free (img);
	}

	return i;
}
