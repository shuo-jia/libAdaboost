#ifndef LINK_LIST_H
#define LINK_LIST_H
#include <stdbool.h>
#include <stdio.h>
#include <stdarg.h>

/**
 * \file link_list.h
 * \brief 一个简单的链表实现 -- 函数声明
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-27
 */
/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
/// 链表结点
struct link_list_node {
	void * data;			///< 链表数据域（存放卫星数据）
	struct link_list_node * next;	///< 链表指针域，指向下一结点
};

/// 链表结构体
struct link_list {
	struct link_list_node head;	///< 链表的头结点
	struct link_list_node * end_ptr;	///< 指向链表最后一个结点
	unsigned int size;		///< 链表长度
};

/// 链表迭代器类型（链表结点）
typedef struct link_list_node * link_iter;

/*******************************************************************************
 * 				    函数原型
 ******************************************************************************/
/**
 * \brief 链表初始化操作
 * \param[out] list 未初始化链表地址，函数执行后链表被初始化
 */
void link_list_init (struct link_list * list);

/**
 * \brief 释放整个链表的内存空间
 * \param[out] list     指向已初始化的链表
 * \param[in] free_data 用于释放卫星数据的回调函数
 * 	（释放对象包括可能的结构体成员及结构体本身）
 */
void link_list_free_full (struct link_list * list, void (*free_data) (void *));

/**
 * \brief 插入数据到链表的起始位置
 * \param[out] list 指向已初始化的链表
 * \param[in] data  待插入数据的地址
 * \return 插入成功返回真，否则返回假
 */
bool link_list_insert (struct link_list * list, void * data);

/**
 * \brief 追加数据到链表末尾
 * \param[out] list 指向已初始化的链表
 * \param[in] data  待插入数据的地址
 * \return 插入成功返回真，否则返回假
 */
bool link_list_append (struct link_list * list, void * data);

/**
* \brief 写入链表到文件
* \param[in] list        已初始化的链表
* \param[out] file       用于保存链表的文件指针
* \param[in] write_data  回调函数，负责将链表数据域指针（第一个参数）指向的内容写
*                        入文件；成功则返回真，否则返回假。
* \param[in, out] ...    可变参数，将被传递给 write_data() 函数，用于实现可重入性
* \return 写入成功则返回真，否则返回假
 */
bool link_list_write (const struct link_list * list, FILE * file,
		bool (*write_data) (const void *, va_list, FILE *), ...);

/**
 * \brief 从文件读取链表
 * \param[out] list     已初始化的空链表
 * \param[in] file      保存有链表数据的文件指针
 * \param[in] read_data 回调函数，返回从文件读取的单个数据；
 *                      成功则返回数据域指针，否则返回 NULL
 * \param[in] ...       可变参数，将被传递给 read_data() 函数，用于实现可重入性
 * \return 读取成功则返回真，否则返回假
 */
bool link_list_read(struct link_list *list, FILE * file,
		    void *(*read_data)(va_list, FILE *), ...);

/**
 * \brief 对链表执行深度复制
 * \param[out] tgt      已初始化但为空的链表（目标链表）
 * \param[in] src       已初始化的链表（源链表）
 * \param[in] copy_data 回调函数，复制单个数据（第一个参数）；成功则返回数据的拷
 *                      贝，否则返回 NULL
 * \param[in] ...       可变参数，将被传递给 copy_data() 函数，用于实现可重入性
 * \return 读取成功则返回 tgt，否则返回 NULL
 */
struct link_list * link_list_copy_full (struct link_list * tgt,
		const struct link_list * src, 
		void * (*copy_data) (const void *, va_list), ...);

/**
 * \brief 移动链表内的元素
 * \param[in, out] tgt_posi 指向待移动结点的前驱结点
 * \param[in, out] src_posi 指向目标位置的前驱结点，待移动结点将被移动到其后
 */
void link_list_move (link_iter tgt_posi, link_iter src_posi);

/**
 * \brief 返回数据域并删除结点
 * \param[out] list  已初始化的链表
 * \param[in] iter   指向目标结点的前驱结点
 * \return 返回迭代器 iter 第一个后继结点的数据域指针，并删除该后继结点
 */
void * link_list_pop (struct link_list *list, link_iter iter);

/**
 * \brief 遍历链表并修改链表元素
 * \param[in, out] list 已初始化的链表
 * \param[in] fun       回调函数，对于链表的每个元素，都将作为参数传递给 fun()
 */
void link_list_traverse (struct link_list *list, void (*fun) (void *));

/**
 * \brief link_list_traverse 的可重入版本，遍历链表并修改链表元素
 * \param[in, out] list 已初始化的链表
 * \param[in] fun       回调函数，对于链表的每个元素，都将作为参数传递给 fun()
 * \param[in, out] ...  可变参数，将被传递给 fun() 函数，用于实现可重入性
 */
void link_list_traverse_r (struct link_list *list,
		void (*fun) (void *, va_list), ...);

/*******************************************************************************
 * 				  内联函数实现
 ******************************************************************************/
/**
 * \brief 获取链表元素个数
 * \param[in] list 已初始化的链表
 * \return 返回链表当前元素个数
 */
static inline unsigned int link_list_size (const struct link_list *list)
{
	return list->size;
}

/**
 * \brief 获取指向头节点的迭代器（头结点不一定保存数据）
 * \param[in] list 已初始化的链表
 * \return 返回迭代器，该迭代器指向头结点
 */
static inline link_iter link_list_head_iter (const struct link_list *list)
{
	return (link_iter) &(list->head);
}

/**
 * \brief 获取指向起始位置的迭代器（头节点后的第一个元素）
 * \param[in] list 已初始化的链表
 * \return 返回迭代器，该迭代器指向头结点后的第一个元素，一般从该元素开始保存数据
 */
static inline link_iter link_list_start_iter (const struct link_list * list)
{
	return list->head.next;
}

/**
 * \brief 获取迭代器当前所指向位置的数据域
 * \param[in] iter 当前迭代器（已初始化）
 * \return 返回链表当前位置的数据域
 */
static inline void * link_list_get_data (const link_iter iter)
{
	return iter->data;
}

/**
 * \brief 将迭代器指向下一位置
 * \param[in, out] iter 已初始化的迭代器
 */
static inline void link_list_next_iter (link_iter * iter)
{
	*iter = (*iter)->next;
}

/**
 * \brief 判断迭代器是否到达超尾
 * \param[in] iter 已初始化的迭代器
 * \return 若迭代器指向超尾位置，返回假；否则返回真
 */
static inline bool link_list_check_end (link_iter iter)
{
	return iter != NULL;
}

#endif
