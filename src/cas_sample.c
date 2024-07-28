#include <stdlib.h>
#include <string.h>
#include "cas_sample.h"
/**
 * \file cas_sample.c
 * \brief Cascade 级联分类器的样本集类型函数实现
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-21
 */

/*******************************************************************************
 *                                    宏函数定义
 ******************************************************************************/
/**
 * \brief 将灰度图像转换为样本
 * \param[out] sp      struct cas_sample *类型，用于保存样本（积分图）
 * \param[in] sp_size  样本尺寸（高或宽）
 * \param[in] i        指示参数 sp 的样本索引
 * \param[in] h        灰度图像的高度
 * \param[in] w        灰度图像的宽度
 * \param[in] img      灰度图像，保存有 h*w 灰度值的数组
 * \param[in] rect_ptr 矩形框，struct cas_rect *类型，矩形框内的灰度图像被处理为
 * 		       样本
 * \param[in] label    此样本的标签，+1 或 - 1
 */
#define IMG_2_SP(sp, sp_size, i, h, w, img, rect_ptr, label)                    \
do{                                                                             \
        img_sampling(sp_size, (void *)sp->X[i], w, (void *)img, rect_ptr);      \
        memcpy(sp->X2[i], sp->X[i], sizeof(sample_t) * sp_size * sp_size);      \
        intgraph(sp_size, sp_size, (void *)sp->X[i]);                           \
        intgraph2(sp_size, sp_size, (void *)sp->X2[i]);                         \
        sp->Y[i] = label;                                                       \
} while (0);

/**
 * \brief 交换两个对象的值
 * \param[in, out] v1 用于交换的对象1
 * \param[in, out] v2 用于交换的对象2
 * \param[in] tmp     临时变量，与 v1、v2 同类型
 */
#define SWAP(v1, v2, tmp)							\
do {										\
	tmp = v1;								\
	v1 = v2;								\
	v2 = tmp;								\
} while(0);

/*******************************************************************************
 * 				  静态函数声明
 ******************************************************************************/
/// 为样本集的成员申请内存空间，但不对内存空间进行初始化
static bool alloc_sample(struct cas_sample *sample, num_t m, imgsz_t img_size);

/**
 * \brief 图片采样。对灰度图片的矩形区域进行采样，将采样结果（缩小后的矩形区域）
 * 	保存至二维数组
 * \param[in] dst_size 目标二维数组大小（行数和列数相同）
 * \param[out] dst     目标二维数组，用于保存采样结果
 * \param[in] src_size 原图片（二维数组）像素列数
 * \param[out] src     原图片（二维数组）
 * \param[in] rect     矩形区域位置及大小
 */
static void img_sampling(imgsz_t dst_size, sample_t dst[][dst_size],
			 imgsz_t src_size, const unsigned char src[][src_size],
			 const struct cas_rect *rect);

/**
 * \brief 随机产生一个矩形框
 * \param[out] rect   矩形框地址，随机产生的矩形框保存于此
 * \param[in] min_len 矩形框的最小边长
 * \param[in] height  矩形框所处图像的高度
 * \param[in] width   矩形框所处图像的宽度
 */
static void rand_rect(struct cas_rect *rect, imgsz_t min_len, imgsz_t height,
		      imgsz_t width);

/**
 * \brief 随机排列样本
 * \param[in, out] sp 已初始化的样本集
 * \param[in] num     样本数量
 */
static void shuffle(struct cas_sample *sp, num_t num);

/**
 * \brief 使用假阳性图片作为样本添加至样本集，直至达到指定样本数量
 * \param[out] sp          已初始化的样本集
 * \param[in, out] index   当前索引。从参数 sp 的第 *index 个样本开始存放样本，
 * 			   函数返回时 *index 置为未存放样本的索引（样本集末尾处）
 * \param[in] m            最大样本数量
 * \param[in] img_size     用作训练的图片尺寸
 * \param[in, out] args    用户自定义的参数，用于保证可重入性
 * \param[in] get_non_face 回调函数，获取非人脸图片，参数 args 将被传递给该函数
 * \param[in] cascade      已训练的级联分类器
 * \param[in] hl           指向 Adaboost 分类器回调函数集
 * \return 成功则返回真，否则返回假
 */
static bool get_remain_samples(struct cas_sample *sp, num_t * index, num_t m,
			       imgsz_t img_size, void *args,
			       cas_non_face_fn get_non_face,
			       const struct cascade *cascade,
			       const struct haar_ada_handles *hl);

/*******************************************************************************
 * 				    函数实现
 ******************************************************************************/
bool init_samples(struct cas_sample *sp, imgsz_t img_size, num_t face,
		  num_t non_face, void *args, cas_face_fn get_face,
		  cas_non_face_fn get_non_face)
{
	if (!alloc_sample(sp, face + non_face, img_size))
		return false;

	num_t index = 0;
	imgsz_t h;			// 图像高度
	imgsz_t w;			// 图像宽度
	num_t id;			// 图像序号
	struct cas_rect rect;		// 人脸框
	const unsigned char *x = NULL;	// 灰度图像
	for (num_t i = 0; i < face; ++i) {
		if ((x = get_face(&h, &w, &rect, args)) == NULL)
			goto err;
		IMG_2_SP(sp, img_size, index, h, w, x, &rect, 1);
		++index;
	}

	for (num_t i = 0; i < non_face; ++i) {
		if ((x = get_non_face(&h, &w, &id, args)) == NULL)
			goto err;
		rand_rect(&rect, img_size, h, w);
		IMG_2_SP(sp, img_size, index, h, w, x, &rect, -1);
		++index;
	}
	shuffle(sp, face + non_face);

	return true;
err:
	free_samples(sp, face + non_face);
	return false;
}

bool update_samples(struct cas_sample *sp, num_t * m, void *args,
		    imgsz_t img_size, cas_non_face_fn get_non_face,
		    const struct cascade *cascade,
		    const struct haar_ada_handles *hl)
{
	num_t i = 0;
	num_t j = 0;
	sample_t *tmp_x;
	label_t tmp_y;

	for (i = 0; i < *m; ++i)
		if (sp->Y[i] > 0) {
			SWAP(sp->X[j], sp->X[i], tmp_x);
			SWAP(sp->X2[j], sp->X2[i], tmp_x);
			SWAP(sp->Y[j], sp->Y[i], tmp_y);
			++j;
		}
	bool status =
	    get_remain_samples(sp, &j, *m - j, img_size, args, get_non_face,
			       cascade, hl);
	num_t new_m = j;
	printf("new_m: %d\n", new_m);
	for (; j < *m; ++j) {
		free(sp->X[j]);
		free(sp->X2[j]);
	}
	*m = new_m;
	shuffle(sp, new_m);
	return status;
}

void free_samples(struct cas_sample *sp, num_t count)
{
	for (num_t i = 0; i < count; ++i) {
		free(sp->X[i]);
		free(sp->X2[i]);
	}
	free(sp->X);
	free(sp->X2);
	free(sp->Y);
}

void intgraph(imgsz_t m, imgsz_t n, sample_t x[m][n])
{
	imgsz_t i, j;
	flt_t line_sum;
	for (j = 1; j < n; ++j)
		x[0][j] += x[0][j - 1];
	for (i = 1; i < m; ++i) {
		line_sum = x[i][0];
		x[i][0] += x[i - 1][0];
		for (j = 1; j < n; ++j) {
			line_sum += x[i][j];
			x[i][j] = x[i - 1][j] + line_sum;
		}
	}
}

void intgraph2(imgsz_t m, imgsz_t n, sample_t x[m][n])
{
	imgsz_t i, j;
	for (i = 0; i < m; ++i)
		for (j = 0; j < n; ++j)
			x[i][j] *= x[i][j];
	intgraph(m, n, x);
}

/*******************************************************************************
 * 				  静态函数实现
 ******************************************************************************/
bool alloc_sample(struct cas_sample *sample, num_t m, imgsz_t img_size)
{
	sample->X = malloc(sizeof(sample_t *) * m);
	sample->X2 = malloc(sizeof(sample_t *) * m);
	sample->Y = malloc(sizeof(label_t) * m);
	if (!sample->X || !sample->X2 || !sample->Y)
		goto malloc_err;
	num_t n;
	size_t len = sizeof(sample_t) * img_size * img_size;
	for (n = 0; n < m; ++n) {
		if (!(sample->X[n] = (sample_t *) malloc(len)))
			goto malloc_arrs_err;
		if (!(sample->X2[n] = (sample_t *) malloc(len))) {
			free(sample->X[n]);
			goto malloc_arrs_err;
		}
	}
	return true;

malloc_arrs_err:
	for (num_t i = 0; i < n; ++i) {
		free(sample->X[i]);
		free(sample->X2[i]);
	}
malloc_err:
	free(sample->X);
	free(sample->X2);
	free(sample->Y);
	return false;
}

void img_sampling(imgsz_t dst_size, sample_t dst[][dst_size], imgsz_t src_size,
		  const unsigned char src[][src_size],
		  const struct cas_rect *rect)
{
	flt_t rate = (flt_t) rect->len / dst_size;

	imgsz_t i, j;
	flt_t posi_i = rect->start_y;
	flt_t posi_j = 0;
	for (i = 0; i < dst_size; ++i) {
		posi_j = rect->start_x;
		for (j = 0; j < dst_size; ++j) {
			dst[i][j] = src[(imgsz_t) posi_i][(imgsz_t) posi_j];
			posi_j += rate;
		}
		posi_i += rate;
	}
}

void rand_rect(struct cas_rect *rect, imgsz_t min_len, imgsz_t height,
	       imgsz_t width)
{
	imgsz_t max_len = width;	// 矩形框的最大边长
	if (height < max_len)
		max_len = height;
	if (max_len == min_len)
		rect->len = min_len;
	else
		rect->len = rand() % (max_len - min_len) + min_len;
	if (width == rect->len)
		rect->start_x = 0;
	else
		rect->start_x = rand() % (width - rect->len);
	if (height == rect->len)
		rect->start_y = 0;
	else
		rect->start_y = rand() % (height - rect->len);
}

void shuffle(struct cas_sample *sp, num_t num)
{
	num_t index;
	sample_t *tmp_x;
	label_t tmp_y;
	for (num_t i = num - 1; i > 0; --i) {
		index = rand() % i;
		SWAP(sp->X[i], sp->X[index], tmp_x);
		SWAP(sp->X2[i], sp->X2[index], tmp_x);
		SWAP(sp->Y[i], sp->Y[index], tmp_y);
	}
}

bool get_remain_samples(struct cas_sample *sp, num_t * index, num_t m,
			imgsz_t img_size, void *args,
			cas_non_face_fn get_non_face,
			const struct cascade *cascade,
			const struct haar_ada_handles *hl)
{
	imgsz_t h;
	imgsz_t w;
	num_t start_id;
	num_t id;
	link_iter iter;
	struct link_list list;
	const struct cas_det_rect * rect;
	const unsigned char * img = get_non_face(&h, &w, &start_id, args);
	do {
		if (img == NULL)
			return false;
		list = cas_detect(cascade, h, w, (void *)img, DETECTOR_DELTA,
				  hl);
		//printf("list len: %d\n", list.size);
		iter = link_list_start_iter(&list);
		while (m > 0 && link_list_check_end(iter)) {
			rect = link_list_get_data(iter);
			IMG_2_SP(sp, img_size, *index, h, w, img, &rect->rect,
				 -1);
			++(*index);
			link_list_next_iter(&iter);
			--m;
		}
		link_list_free_full(&list, free);
		img = get_non_face(&h, &w, &id, args);
	} while(m > 0 && id != start_id);
	return true;
}
