#include <sys/param.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>
#include "cascade.h"
/**
 * \file cascade.c
 * \brief 级联的(Cascade) adaboost 分类器函数定义
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-15
 */

/*******************************************************************************
 * 				  静态函数声明
 ******************************************************************************/
/// Adaboost 写入方法的包装函数，用作链表的回调函数
static bool ada_write(const void *adaboost, va_list ap, FILE * file);
/// Adaboost 读取方法的包装函数，用作链表的回调函数
static void *ada_read(va_list ap, FILE * file);
/// Adaboost 内存释放方法的包装函数，用作链表的回调函数
static void ada_free(void *adaboost, va_list ap);

/// 根据训练结果更新调整训练集和验证集。
/** 在验证集中留下判别为阳性的样本；在训练集留下阳性样本及假阳性样本，并随机生
 * 成足量的样本 */
static void update_samples(num_t * l, num_t * m, imgsz_t n,
			   const sample_t * X[], const sample_t * X2[],
			   label_t Y[], const struct haar_adaboost *adaboost,
			   const struct haar_ada_handles *hl);

/// 使用极大值抑制方法处理重叠边框
static void NMS(struct link_list *list, flt_t threshold);

/// 返回一个一维数组，元素为 0, 1, ..., m-1 的随机排列；失败则返回 NULL
/** 返回的数组可用 free() 函数释放内存 */
static num_t *randperm(num_t m);

/*******************************************************************************
 * 				    函数实现
 ******************************************************************************/
bool cas_train(struct cascade *cascade, flt_t d, flt_t f, flt_t F, num_t l,
	       num_t m, imgsz_t n, sample_t * const X[], sample_t * const X2[],
	       const label_t Y[], const struct haar_ada_handles *hl)
{
	// Adaboost 地址
	struct haar_adaboost *adaboost = NULL;
	flt_t ada_f_p_ratio;	// AdaBoost 假阳率
	flt_t ada_det_ratio;	// AdaBoost 检测率

	cascade->img_size = n;	// 设置图像尺寸
	cascade->f_p_ratio = 1;	// 当前假阳率
	cascade->det_ratio = 1;	// 当前检测率
	link_list_init(&cascade->adaboost);

	label_t *Y_cp = malloc(sizeof(label_t) * (l + m));
	const sample_t **X_cp = malloc(sizeof(sample_t *) * (l + m));
	const sample_t **X2_cp = malloc(sizeof(sample_t) * (l + m));
	if (Y_cp == NULL || X_cp == NULL || X2_cp == NULL)
		goto cp_err;
	else {
		num_t *ids = randperm(l + m);
		if (ids == NULL)
			goto cp_err;
		for (num_t i = 0; i < l + m; ++i) {
			X_cp[i] = X[ids[i]];
			X2_cp[i] = X2[ids[i]];
			Y_cp[i] = Y[ids[i]];
		}
		free(ids);
	}
#ifdef LOG
	printf("Training start.\n");
#endif
	while (cascade->f_p_ratio > F) {
		ada_det_ratio = d;
		ada_f_p_ratio = f;
		if ((adaboost = malloc(sizeof(struct haar_adaboost))) == NULL)
			goto new_ab_err;
		if (!hl->train(adaboost, &ada_det_ratio, &ada_f_p_ratio, l, m,
			       n, n, X_cp, X2_cp, Y_cp, &hl->wl_hl))
			goto train_ab_err;
		if (!link_list_append(&cascade->adaboost, adaboost))
			goto train_ab_err;
		// 更新当前假阳率、检测率
		cascade->f_p_ratio *= ada_f_p_ratio;
		cascade->det_ratio *= ada_det_ratio;
#ifdef LOG
		printf("Current detection ratio: %f\n", cascade->det_ratio);
		printf("Current false positive ratio: %f\n",
		       cascade->f_p_ratio);
		printf("Target maximum false positive ratio: %f\n", F);
#endif
		if (ada_f_p_ratio > f)
			break;
		// 调整样本
		update_samples(&l, &m, n, X_cp, X2_cp, Y_cp, adaboost, hl);
	}
#ifdef LOG
	printf("Training end.\n");
#endif

	free(Y_cp);
	free(X_cp);
	free(X2_cp);
	return true;

train_ab_err:
	free(adaboost);
new_ab_err:
cp_err:
	free(Y_cp);
	free(X_cp);
	free(X2_cp);
	return false;
}

bool cas_cat(struct cascade *dst, struct cascade *src)
{
	if (dst->img_size != src->img_size)
		return false;

	dst->adaboost.end_ptr->next = src->adaboost.head.next;
	dst->adaboost.end_ptr = src->adaboost.end_ptr;
	dst->adaboost.size += src->adaboost.size;

	link_list_init(&src->adaboost);
	dst->det_ratio *= src->det_ratio;
	dst->f_p_ratio *= src->f_p_ratio;
	return true;
}

bool cas_write(const struct cascade *cascade, FILE * file,
	       const struct haar_ada_handles *hl)
{
	// 写入训练参数 d、f、F、训练图像大小
	if (fwrite(&cascade->img_size, sizeof(imgsz_t), 1, file) < 1)
		return false;
	if (fwrite(&cascade->f_p_ratio, sizeof(flt_t), 1, file) < 1)
		return false;
	if (fwrite(&cascade->det_ratio, sizeof(flt_t), 1, file) < 1)
		return false;

	// 写入 AdaBoost 分类器
	return link_list_write(&cascade->adaboost, file, ada_write, hl);
}

bool cas_read(struct cascade *cascade, FILE * file,
	      const struct haar_ada_handles *hl)
{
	// 读取训练参数 d、f、F
	if (fread(&cascade->img_size, sizeof(imgsz_t), 1, file) < 1)
		return false;
	if (fread(&cascade->f_p_ratio, sizeof(flt_t), 1, file) < 1)
		return false;
	if (fread(&cascade->det_ratio, sizeof(flt_t), 1, file) < 1)
		return false;

	// 读取 AdaBoost 分类器
	link_list_init(&cascade->adaboost);
	return link_list_read(&cascade->adaboost, file, ada_read, hl);
}

void cas_free(struct cascade *cascade, const struct haar_ada_handles *hl)
{
	link_list_traverse_r(&cascade->adaboost, ada_free, hl);
	link_list_free_full(&cascade->adaboost, free);
}

void cas_intgraph(imgsz_t m, imgsz_t n, flt_t x[m][n])
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

void cas_intgraph2(imgsz_t m, imgsz_t n, flt_t x[m][n])
{
	imgsz_t i, j;
	for (i = 0; i < m; ++i)
		for (j = 0; j < n; ++j)
			x[i][j] *= x[i][j];
	cas_intgraph(m, n, x);
}

flt_t IoU(const struct cas_rect *rect1, const struct cas_rect *rect2)
{
	struct cas_rect intersection;
	intersection.start_x = MAX(rect1->start_x, rect2->start_x);
	intersection.start_y = MAX(rect1->start_y, rect2->start_y);
	intersection.width = MIN(rect1->start_x + rect1->width,
				 rect2->start_x + rect2->width)
	    - intersection.start_x;
	intersection.height = MIN(rect1->start_y + rect1->height,
				  rect2->start_y + rect2->height)
	    - intersection.start_y;
	if (intersection.width < 0 || intersection.height < 0)
		return 0;
	flt_t s1 = rect1->height * rect1->width;
	flt_t s2 = rect2->height * rect2->width;
	flt_t s = intersection.height * intersection.width;
	return (flt_t) s / (s1 + s2 - s);
}

flt_t cas_h(const struct cascade *cascade, imgsz_t n, imgsz_t wid,
	    const flt_t x[n][wid], const flt_t x2[n][wid],
	    const struct haar_ada_handles *hl)
{
	flt_t scale = (flt_t) n / cascade->img_size;
	flt_t result;
	struct haar_adaboost *adaboost = NULL;
	link_iter iter = link_list_start_iter(&cascade->adaboost);
	while (link_list_check_end(iter)) {
		adaboost = link_list_get_data(iter);
		if ((result = hl->h(adaboost, n, n, wid, x, x2, scale,
				    &hl->wl_hl)) < 0)
			return result;
		link_list_next_iter(&iter);
	}
	return result;
}

flt_t cas_nextobj(const struct cascade *cascade, struct cas_rect *rect,
		  imgsz_t * delta, imgsz_t h, imgsz_t w, const flt_t x[][w],
		  const flt_t x2[][w], const struct haar_ada_handles *hl)
{
	flt_t result;
	const flt_t scale_times = 1.25;
	const void *x_start = NULL;
	const void *x2_start = NULL;

	imgsz_t min_size = (h > w) ? w : h;
	rect->start_x += *delta;
	while (rect->height < min_size) {
		while (rect->start_y <= h - rect->height) {
			while (rect->start_x <= w - rect->width) {
				x_start = &x[rect->start_y][rect->start_x];
				x2_start = &x2[rect->start_y][rect->start_x];
				result =
				    cas_h(cascade, rect->height, w, x_start,
					  x2_start, hl);
				if (result > 0)
					return result;
				rect->start_x += *delta;
			}
			rect->start_x = 0;
			rect->start_y += *delta;
		}
		rect->start_y = 0;
		rect->height *= scale_times;
		rect->width = rect->height;
		*delta *= scale_times;
	}

	return -1;
}

struct link_list cas_detect(const struct cascade *cascade, imgsz_t h,
			    imgsz_t w, unsigned char img[h][w], imgsz_t delta,
			    const struct haar_ada_handles *hl)
{
	imgsz_t i, j;
	struct link_list list;
	struct cas_det_rect *rect_ptr = NULL;
	struct cas_det_rect rect =
	    { { 0, 0, cascade->img_size, cascade->img_size }, 0 };
	flt_t x[h][w];
	flt_t x2[h][w];

	for (i = 0; i < h; ++i)
		for (j = 0; j < w; ++j) {
			x[i][j] = img[i][j];
			x2[i][j] = img[i][j];
		}
	cas_intgraph(h, w, x);
	cas_intgraph2(h, w, x2);

	// 构建一个由所有可能含有目标的边框构成的链表
	link_list_init(&list);
	while ((rect.confidence = cas_nextobj(cascade, &rect.rect, &delta,
					      h, w, x, x2, hl)) > 0) {
		rect_ptr = malloc(sizeof(struct cas_det_rect));
		if (rect_ptr == NULL)
			return list;
		*rect_ptr = rect;
		if (!link_list_append(&list, rect_ptr)) {
			free(rect_ptr);
			return list;
		}
	}
	// 删除重叠窗口
	NMS(&list, 0.1);
	return list;
}

/*******************************************************************************
 * 				  静态函数实现
 ******************************************************************************/
bool ada_write(const void *adaboost, va_list ap, FILE * file)
{
	const struct haar_ada_handles *hl = va_arg(ap,
						   struct haar_ada_handles *);
	return hl->write(adaboost, file, &hl->wl_hl);
}

void *ada_read(va_list ap, FILE * file)
{
	const struct haar_ada_handles *hl = va_arg(ap,
						   struct haar_ada_handles *);
	struct haar_adaboost *ada = malloc(sizeof(struct haar_adaboost));
	if (ada == NULL)
		return NULL;
	if (!hl->read(ada, file, &hl->wl_hl)) {
		free(ada);
		return NULL;
	}
	return ada;
}

void ada_free(void *adaboost, va_list ap)
{
	const struct haar_ada_handles *hl = va_arg(ap,
						   struct haar_ada_handles *);
	hl->free(adaboost, &hl->wl_hl);
}

// 在前 l 个样本（验证集）中保留分类结果为阳性的样本（真阳性和假阳性）
// 在后 m 个样本（训练集）中保留阳性样本和假阳性样本
void update_samples(num_t * l, num_t * m, imgsz_t n, const sample_t * X[],
		    const sample_t * X2[], label_t Y[],
		    const struct haar_adaboost *adaboost,
		    const struct haar_ada_handles *hl)
{
	num_t train_i;		// 训练集索引
	num_t val_i;		// 验证集索引
	num_t j = 0;		// 当前索引

	for (val_i = 0; val_i < *l; ++val_i)
		if (hl->h(adaboost, n, n, n, (void *)X[val_i],
			  (void *)X2[val_i], 1, &hl->wl_hl) > 0) {
			X[j] = X[val_i];
			X2[j] = X2[val_i];
			Y[j] = Y[val_i];
			++j;
		}
	*l = j;
	for (train_i = val_i; train_i < val_i + *m; ++train_i)
		if (Y[train_i] > 0 ||
		    hl->h(adaboost, n, n, n, (void *)X[train_i],
			  (void *)X2[train_i], 1, &hl->wl_hl) > 0) {
			X[j] = X[train_i];
			X2[j] = X2[train_i];
			Y[j] = Y[train_i];
			++j;
		}
	*m = j - *l;
}

// 极大值抑制方法（NMS）处理重叠窗口
void NMS(struct link_list *list, flt_t threshold)
{
	flt_t iou;		// 窗口重叠度
	struct cas_det_rect *rect1, *rect2;
	struct cas_det_rect *rect_ptr = NULL;
	flt_t max_val;
	link_iter iter;
	link_iter prev;
	link_iter max_posi;	// 最大置信度窗口的前一节点
	link_iter head;		// 未处理窗口的头节点
	head = link_list_head_iter(list);
	while (head->next != NULL) {
		// 查找最大置信度边框
		iter = head;
		prev = head;
		max_val = -DBL_MAX;
		max_posi = NULL;
		link_list_next_iter(&iter);
		while (link_list_check_end(iter)) {
			rect_ptr = link_list_get_data(iter);
			if (rect_ptr->confidence > max_val) {
				max_val = rect_ptr->confidence;
				max_posi = prev;
			}
			prev = iter;
			link_list_next_iter(&iter);
		}
		// 最大置信度窗口移动到头部
		link_list_move(head, max_posi);
		link_list_next_iter(&head);
		// 删除重叠度大于给定阈值的窗口
		iter = head;
		prev = head;
		link_list_next_iter(&iter);
		while (link_list_check_end(iter)) {
			rect1 = link_list_get_data(head);
			rect2 = link_list_get_data(iter);
			iou = IoU(&rect1->rect, &rect2->rect);
			if (iou > threshold) {
				link_list_next_iter(&iter);
				rect_ptr = link_list_pop(list, prev);
				free(rect_ptr);
			} else {
				prev = iter;
				link_list_next_iter(&iter);
			}
		}
	}
}

num_t *randperm(num_t m)
{
	num_t *arr = malloc(sizeof(num_t) * m);
	if (arr == NULL)
		return NULL;
	for (num_t i = 0; i < m; ++i)
		arr[i] = i;

	// 随机排列（随机交换）
	num_t tmp;
	num_t index;
	for (num_t i = m - 1; i > 0; --i) {
		index = rand() % i;
		tmp = arr[i];
		arr[i] = arr[index];
		arr[index] = tmp;
	}

	return arr;
}
