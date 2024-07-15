#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include "sample.h"
#include "pretrain.h"
#include "image.h"

/*******************************************************************************
 * 				   宏常量定义
 ******************************************************************************/
// 训练集占比（其余设置为验证集）
#define TRAIN_SET_PERCENT 0.7
// 每个 AdaBoost 分类器的检测率、假阳率
#define DET_RATE 0.995
#define FP_RATE 0.8
// 最大假阳率
#define MAX_FP_RATE 1E-5
// 窗口滑动间隔
#define DELTA 2
// 最大负例样本同正例样本之比
#define NEG_PER_POSI 20

/*******************************************************************************
 * 				  静态函数声明
 ******************************************************************************/
// 对前 n 张图片进行测试，保存假阳性样本，并返回错误率
// 如出现错误，则返回负数
static flt_t save_false_posi (struct cascade * pcascade, num_t n, FILE * p_mark,
		FILE * n_mark, const char * dir_path,
		const struct haar_ada_handles * hl);

// 链表的回调函数，对每个矩形框进行检测，更新假阳性的矩形框数量，并将假阳性的
// 矩形框写入文件
static void rect_check (void * v_rect, va_list ap);

/*******************************************************************************
 * 				    函数定义
 ******************************************************************************/
bool pre_train (struct cascade * pcascade, num_t max_posi, imgsz_t face_size,
		flt_t err_rate, const char * posi_mark, const char * neg_mark,
		const char * dir_path, const struct haar_ada_handles *hl)
{
	struct sample sample;
	FILE * p_mark = fopen (posi_mark, "r");
	if (p_mark == NULL)
		return false;
	FILE * n_mark = fopen (neg_mark, "w+");
	if (n_mark == NULL)
		goto open_err;

	if (!get_rand_sample (&sample, max_posi, NEG_PER_POSI, face_size,
				p_mark, n_mark, dir_path))
		goto make_sp_err;
	num_t m = sample.m * TRAIN_SET_PERCENT;		// 训练集长度
	num_t l = sample.m - m;				// 验证集长度

	if (!cas_train(pcascade, DET_RATE, FP_RATE, MAX_FP_RATE, l, m,
			face_size, sample.X, sample.X2, sample.Y, hl))
		goto train_err;
	free_sample (&sample);
	// 保存当前位置，以便读取新阴性样本
	fpos_t pos;
	fgetpos (n_mark, &pos);
	struct cascade new_cas;
	while (save_false_posi (pcascade, max_posi, p_mark, n_mark, dir_path,
				hl) > err_rate) {
		rewind (p_mark);
		fsetpos (n_mark, &pos);
		// 从上次训练得到的假阳性样本构建样本集
		if (!get_mark_sample (&sample, max_posi, face_size, p_mark,
					n_mark, dir_path))
			goto make_sp_err;
		if (sample.m < 2 * max_posi) {
			free_sample (&sample);
			break;
		}
		m = sample.m * TRAIN_SET_PERCENT;
		l = sample.m - m;
		if (!cas_train(&new_cas, DET_RATE, FP_RATE, MAX_FP_RATE, l, m,
				face_size, sample.X, sample.X2, sample.Y, hl))
			goto train_err;
		if (!cas_cat (pcascade, &new_cas)) {
			cas_free (&new_cas, hl);
			goto train_err;
		}
		free_sample (&sample);
		fgetpos (n_mark, &pos);
	}

	fclose (p_mark);
	fclose (n_mark);
	return true;

train_err:
	free_sample (&sample);
make_sp_err:
	fclose (n_mark);
open_err:
	fclose (p_mark);
	return false;
}

/*******************************************************************************
 * 				  静态函数定义
 ******************************************************************************/
flt_t save_false_posi (struct cascade * pcascade, num_t n, FILE * p_mark,
		FILE * n_mark, const char * dir_path,
		const struct haar_ada_handles * hl)
{
	char filename [MAX_FILENAME];
	strcpy (filename, dir_path);
	char *f_ptr = filename + strlen (filename);

	flt_t err = 0;
	num_t err_count;
	struct rectangle rect;
	struct image *img = NULL;
	struct link_list list;
	rewind (p_mark);
	for (num_t i = 0; i < n; ++i) {
		err_count = 0;
		if (fscanf (p_mark, "%s %d %d %d %d", f_ptr, &rect.start_x,
			&rect.start_y, &rect.height, &rect.width) < 5)
			return -1;
		if ((img = imread_pgm (filename)) == NULL)
			return -1;
		list = cas_detect (pcascade, img->height, img->width,
				(void *)(img->img), DELTA, hl);
		link_list_traverse_r (&list, rect_check, &err_count, f_ptr, n_mark, &rect);
		if (err_count > 0)
			++err;
		link_list_free_full (&list, free);
		free (img);
	}
	err /= n;
#ifdef LOG
	printf ("Err: %f\n", err);
#endif
	return err;
}

void rect_check (void * v_rect, va_list ap)
{
	num_t * err_count = va_arg(ap, num_t *);
	const char * f_ptr = va_arg(ap, const char *);
	FILE * n_mark = va_arg(ap, FILE *);
	const struct rectangle * face = va_arg(ap, struct rectangle *);

	struct cas_rect rect1, rect2;
	rect_cast_r (&rect1, v_rect);
	rect_cast_r (&rect2, face);
	if (IoU (&rect1, &rect2) > 0.7)
		return;

	// 假阳性样本处理
	const struct rectangle * rect = v_rect;
	++(*err_count);
	fprintf (n_mark, "%s %d %d %d %d\n", f_ptr, rect->start_x, rect->start_y,
			rect->height, rect->width);
}
