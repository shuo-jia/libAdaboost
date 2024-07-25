#include <stdlib.h>
#include "stump_ga_base.h"
#include "haar_stump_pvt.h"
#include "haar_stump_ga.h"

/**
 * \file haar_stump_ga.c
 * \brief haar_stump 训练函数重载，使用进化算法寻优（函数实现）
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */
/*******************************************************************************
 * 				   宏函数定义
 ******************************************************************************/
/// 产生 [a, b] 范围内的随机整数（a、b 均为整数且 a < b）
#define RAND_RANGE(a, b) ((a) + rand() % ((b) - (a) + 1))

/**
 * \brief 获取 Haar 特征矩形宽度的上界
 * \param[in] args 实际类型为 struct sp_wrap *
 * \param[in] ind 实际类型为 struct haar_feature *
 * \return 返回 Haar 特征 ind 在当前条件下矩形宽度的上界
 */
#define UB_WIDTH(args, ind) (((args)->w - 1) / rect_x_ct[(ind)->type])

/**
 * \brief 获取 Haar 特征矩形高度的上界
 * \param[in] args 实际类型为 struct sp_wrap *
 * \param[in] ind 实际类型为 struct haar_feature *
 * \return 返回 Haar 特征 ind 在当前条件下矩形高度的上界
 */
#define UB_HEIGHT(args, ind) (((args)->h - 1) / rect_y_ct[(ind)->type])

/**
 * \brief 获取 Haar 特征矩形左上角横坐标的上界
 * \param[in] args 实际类型为 struct sp_wrap *
 * \param[in] ind 实际类型为 struct haar_feature *
 * \return 返回 Haar 特征 ind 在当前条件下矩形左上角横坐标的上界
 */
#define UB_STARTX(args, ind)							\
	((args)->w - (ind)->width * rect_x_ct[(ind)->type] - 1)

/**
 * \brief 获取 Haar 特征矩形左上角纵坐标的上界
 * \param[in] args 实际类型为 struct sp_wrap *
 * \param[in] ind  实际类型为 struct haar_feature *
 * \return 返回 Haar 特征 ind 在当前条件下矩形左上角纵坐标的上界
 */
#define UB_STARTY(args, ind)							\
	((args)->h - (ind)->height * rect_y_ct[(ind)->type] - 1)

/**
 * \brief 对两个父代个体的某个成员进行交叉，并赋给子代个体的对应成员
 * \param[out] ind     子代个体
 * \param[in] p1       父代个体之一
 * \param[in] p2       父代个体之二
 * \param[in] member   成员名，可以是 width、height、start_x 或 start_y
 * \param[in] ub_macro 上界获取方法，即以"UB_"为前缀的宏
 * \param[in] args     struct sp_wrap * 类型的参数
 */
#define CROSS(ind, p1, p2, member, ub_macro, args)				\
	do {									\
		imgsz_t ub = ub_macro(args, ind);				\
		flt_t r = (flt_t) rand() / RAND_MAX;				\
		(ind)->member = (1-r) * (p1)->member + r * (p2)->member + 0.5;	\
		if ((ind)->member > ub)						\
			(ind)->member = ub;					\
	} while(0);

/**
 * \brief 对个体的某个成员进行变异操作
 * \param[out] ind     子代个体
 * \param[in] member   成员名，可以是 width、height、start_x 或 start_y
 * \param[in] lb       指定成员的下界
 * \param[in] ub_macro 上界获取方法，即以"UB_"为前缀的宏
 * \param[in] args     struct sp_wrap * 类型的参数
 * \param[in] step     变异步长
 */
#define MUTATE(ind, member, lb, ub_macro, args, step)				\
	do {									\
		imgsz_t ub = ub_macro(args, ind);				\
		flt_t r = ((flt_t) rand() / RAND_MAX - 0.5) * 2 * step;		\
		if ((ind)->member + r > ub)					\
			(ind)->member = ub;					\
		else if ((ind)->member < lb - r)				\
			(ind)->member = lb;					\
		else								\
			(ind)->member += r;					\
	} while(0);

/*******************************************************************************
 * 				  静态函数声明
 ******************************************************************************/
/// 进化算法回调函数：初始化单个个体
static void ga_init(void *individual, const void *samples);
/// 进化算法回调函数：交叉产生单个后代
static void ga_crossover(void *child, const void *parent1,
			 const void *parent2, const void *samples);
/// 进化算法回调函数：对单个个体进行变异操作
static void ga_mutate(void *individual, const void *samples);
/// 进化算法回调函数：对样本集包装结构体、回调函数集进行初始化
static bool init_setting(struct sp_wrap *sp, struct stump_ga_handles *hl,
			 num_t m, const sample_t * const *X,
			 const sample_t * const *X2, imgsz_t h, imgsz_t w);
/// 释放内存空间
static inline void free_setting(struct sp_wrap *sp);

/*******************************************************************************
 * 				    函数实现
 ******************************************************************************/
bool haar_stump_ga_train(void *stump, num_t m, imgsz_t h, imgsz_t w,
			 const sample_t * const X[], const sample_t * const X2[],
			 const label_t Y[], const flt_t D[])
{
	struct sp_wrap sp;
	struct stump_ga_handles hl;
	if (!init_setting(&sp, &hl, m, X, X2, h, w))
		return false;

	struct haar_stump *ptr = stump;
	if (!cstump_ga(&ptr->base, &ptr->feature, sizeof(struct haar_feature),
		       m, &sp, Y, D, &hl)) {
		free_setting(&sp);
		return false;
	}

	free_setting(&sp);
	return true;
}

bool haar_stump_ga_cf_train(void *stump, num_t m, imgsz_t h, imgsz_t w,
			    const sample_t * const X[],
			    const sample_t * const X2[], const label_t Y[],
			    const flt_t D[])
{
	struct sp_wrap sp;
	struct stump_ga_handles hl;
	if (!init_setting(&sp, &hl, m, X, X2, h, w))
		return false;

	struct haar_stump_cf *ptr = stump;
	if (!cstump_cf_ga
	    (&ptr->base, &ptr->feature, sizeof(struct haar_feature), m, &sp, Y,
	     D, &hl)) {
		free_setting(&sp);
		return false;
	}

	free_setting(&sp);
	return true;
}

/*******************************************************************************
 * 				  静态函数定义
 ******************************************************************************/
void ga_init(void *individual, const void *samples)
{
	struct haar_feature *ind = individual;
	const struct sp_wrap *sp = samples;
	ind->type = RAND_RANGE(FEAT_START + 1, FEAT_END - 1);
	ind->width = RAND_RANGE(1, UB_WIDTH(sp, ind));
	ind->height = RAND_RANGE(1, UB_HEIGHT(sp, ind));
	ind->start_x = RAND_RANGE(0, UB_STARTX(sp, ind));
	ind->start_y = RAND_RANGE(0, UB_STARTY(sp, ind));
}

void ga_crossover(void *child, const void *parent1, const void *parent2,
		  const void *samples)
{
	struct haar_feature *cld = child;
	const struct haar_feature *prt1 = parent1;
	const struct haar_feature *prt2 = parent2;
	const struct sp_wrap *sp = samples;

	cld->type = ((rand() % 2) == 0) ? prt1->type : prt2->type;
	CROSS(cld, prt1, prt2, width, UB_WIDTH, sp);
	CROSS(cld, prt1, prt2, height, UB_HEIGHT, sp);
	CROSS(cld, prt1, prt2, start_x, UB_STARTX, sp);
	CROSS(cld, prt1, prt2, start_y, UB_STARTY, sp);
}

void ga_mutate(void *individual, const void *samples)
{
	struct haar_feature *ind = individual;
	const struct sp_wrap *sp = samples;
	const float step = sp->h / 4.0;

	ind->type = RAND_RANGE(FEAT_START + 1, FEAT_END - 1);
	MUTATE(ind, width, 1, UB_WIDTH, sp, step);
	MUTATE(ind, height, 1, UB_HEIGHT, sp, step);
	MUTATE(ind, start_x, 0, UB_STARTX, sp, step);
	MUTATE(ind, start_y, 0, UB_STARTY, sp, step);
}

bool init_setting(struct sp_wrap *sp, struct stump_ga_handles *hl, num_t m,
		  const sample_t * const *X, const sample_t * const *X2,
		  imgsz_t h, imgsz_t w)
{
	sp->X = X;
	sp->X2 = X2;
	sp->h = h;
	sp->w = w;
	sp->vector = malloc(sizeof(sample_t) * m);
	if (sp->vector == NULL)
		return false;

	hl->gen = GEN;
	hl->m = POP_SIZE;
	hl->p_c = P_C;
	hl->p_m = P_M;
	hl->init = ga_init;
	hl->crossover = ga_crossover;
	hl->mutate = ga_mutate;
	hl->get_vals = get_vals_raw;
	hl->update_opt = update_opt;
	return true;
}

void free_setting(struct sp_wrap *sp)
{
	free(sp->vector);
}
