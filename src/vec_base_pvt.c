#include "vec_base_pvt.h"
/**
 * \file vec_base_pvt.c
 * \brief 样本集为向量集时共用的函数实现。
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */

/*******************************************************************************
 * 				   宏函数定义
 ******************************************************************************/
/**
 * \brief 弱学习器数组读写函数
 * \param[in, out] wl          弱学习器数组地址
 * \param[in] nmemb            元素数量
 * \param[in, out] file        FILE* 类型
 * \param[in] hl               struct wl_handles *类型
 * \param[in] rw_fun           fread 或 fwrite
 * \param[in] hl_fun           read 或 write
 * \return 返回成功读/写的弱学习器数量
 */
#define WL_RW(wl, nmemb, file, hl, rw_fun, hl_fun)				\
({										\
 	turn_t count = 0;							\
	do {									\
		if (hl->hl_fun == NULL) {					\
			if (rw_fun(wl, hl->size, nmemb, file) < nmemb)		\
				break;						\
			count = nmemb;						\
		}								\
		else {								\
			for (; count < nmemb; ++count, wl += hl->size)		\
				if (!hl->hl_fun (wl, file))			\
					break;					\
		}								\
	} while (0);								\
	count;									\
})

/*******************************************************************************
 * 				    函数定义
 ******************************************************************************/
turn_t vec_wl_read(unsigned char *weaklearner, turn_t nmemb,
		   const struct wl_handles *handles, FILE * file)
{
	return WL_RW(weaklearner, nmemb, file, handles, fread, read);
}

turn_t vec_wl_write(const unsigned char *weaklearner, turn_t nmemb,
		    const struct wl_handles *handles, FILE * file)
{
	return WL_RW(weaklearner, nmemb, file, handles, fwrite, write);
}

turn_t vec_wl_copy(unsigned char *dst, const unsigned char *src,
		   turn_t nmemb, const struct wl_handles *handles)
{
	turn_t count;
	if (handles->copy != NULL) {
		for (count = 0; count < nmemb; count++) {
			if (!handles->copy(dst, src))
				break;
			dst += handles->size;
			src += handles->size;
		}
	} else {
		memcpy(dst, src, nmemb * handles->size);
		count = nmemb;
	}
	return count;
}

void vec_wl_free(unsigned char *wl, turn_t nmemb,
		 const struct wl_handles *handles)
{
	if (handles->free == NULL)
		return;
	for (turn_t i = 0; i < nmemb; ++i) {
		handles->free(wl);
		wl += handles->size;
	}
}
