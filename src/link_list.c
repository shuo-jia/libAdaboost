#include <stdbool.h>
#include <stdlib.h>
#include "link_list.h"

/**
 * \file link_list.c
 * \brief 一个简单的链表实现 -- 函数实现
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-27
 */
/*******************************************************************************
 * 				  静态函数声明
 ******************************************************************************/
/// 删除 node 结点之后的所有结点（不包括 node），并释放卫星数据的内存空间
static void free_nodes (struct link_list_node * node, void (*free_data) (void *));

/*******************************************************************************
 * 				    函数定义
 ******************************************************************************/
// 初始化操作
void link_list_init (struct link_list * list)
{
	list->head.data = NULL;
	list->head.next = NULL;
	list->end_ptr = &list->head;
	list->size = 0;
}

// 释放整个链表的内存空间
void link_list_free_full (struct link_list * list, void (*free_data) (void *))
{
	free_nodes (&list->head, free_data);
	list->end_ptr = &list->head;
	list->size = 0;
}

// 插入数据到开始位置
bool link_list_insert (struct link_list * list, void * data)
{
	struct link_list_node * node = malloc (sizeof(struct link_list_node));
	if (node == NULL)
		return false;
	node->data = data;
	node->next = list->head.next;
	list->head.next = node;
	++list->size;

	return true;
}

// 追加数据到末尾
bool link_list_append (struct link_list * list, void * data)
{
	struct link_list_node * node = malloc (sizeof(struct link_list_node));
	if (node == NULL)
		return false;
	node->data = data;
	node->next = NULL;
	list->end_ptr->next = node;
	list->end_ptr = node;
	++list->size;

	return true;
}

// 写入链表到文件
bool link_list_write (const struct link_list * list, FILE * file,
		bool (*write_data) (const void *, va_list, FILE *), ...)
{
	if (write_data == NULL)
		return false;
	if (fwrite (&list->size, sizeof(unsigned int), 1, file) < 1)
		return false;

	va_list ap;
	const struct link_list_node * ptr = &list->head;
	while (ptr->next != NULL) {
		ptr = ptr->next;
		va_start (ap, write_data);
		if (! write_data (ptr->data, ap, file)) {
			va_end(ap);
			return false;
		}
		va_end(ap);
	}

	return true;
}

bool link_list_read (struct link_list * list, FILE * file,
		void * (*read_data) (va_list, FILE *), ...)
{
	unsigned int i;
	unsigned int tmp_size;
	if (read_data == NULL)
		return false;

	if (fread (&tmp_size, sizeof(unsigned int), 1, file) < 1)
		return false;

	va_list ap;
	link_iter iter = &list->head;
	for (i = 0; i < tmp_size; ++i) {
		if (! link_list_append (list, NULL))
			goto append_err;
		va_start (ap, read_data);
		if ((list->end_ptr->data = read_data(ap, file)) == NULL)
			goto read_err;
		iter = list->end_ptr;
		va_end (ap);
	}

	return true;
read_err:
	link_list_pop (list, iter);
	va_end (ap);
append_err:
	return false;
}

struct link_list * link_list_copy_full (struct link_list * tgt,
		const struct link_list * src, 
		void * (*copy_data) (const void *, va_list), ...)
{
	const struct link_list_node * src_ptr = &src->head;
	if (copy_data == NULL)
		return NULL;

	va_list ap;
	link_iter iter = &tgt->head;
	while (src_ptr->next != NULL) {
		src_ptr = src_ptr->next;
		if (! link_list_append (tgt, NULL))
			goto append_err;
		va_start (ap, copy_data);
		if ((tgt->end_ptr->data = copy_data(src_ptr->data, ap)) == NULL)
			goto copy_err;
		va_end (ap);
		iter = tgt->end_ptr;
	}
	return tgt;
copy_err:
	va_end (ap);
	link_list_pop (tgt, iter);
append_err:
	return NULL;
}

void link_list_move (link_iter tgt_posi, link_iter src_posi)
{
	if (src_posi->next == NULL)
		return;

	struct link_list_node *tmp = src_posi->next;
	src_posi->next = src_posi->next->next;
	tmp->next = tgt_posi->next;
	tgt_posi->next = tmp;
}

void * link_list_pop (struct link_list *list, link_iter iter)
{
	if (iter->next == NULL)
		return NULL;

	struct link_list_node *tmp = iter->next;
	void * data = tmp->data;
	iter->next = iter->next->next;
	free (tmp);

	--list->size;
	if (iter->next == NULL)
		list->end_ptr = iter;
	return data;
}

void link_list_traverse (struct link_list *list, void (*fun) (void *))
{
	link_iter iter = link_list_start_iter (list);
	while (link_list_check_end (iter)) {
		fun (link_list_get_data (iter));
		link_list_next_iter (&iter);
	}
}

void link_list_traverse_r (struct link_list *list,
		void (*fun) (void *, va_list), ...)
{
	va_list ap;
	link_iter iter = link_list_start_iter (list);
	while (link_list_check_end (iter)) {
		va_start (ap, fun);
		fun (link_list_get_data (iter), ap);
		va_end (ap);
		link_list_next_iter (&iter);
	}
}

/*******************************************************************************
 * 				  静态函数定义
 ******************************************************************************/
void free_nodes (struct link_list_node * node, void (*free_data) (void *))
{
	struct link_list_node * tmp;
	while (node->next != NULL) {
		tmp = node->next;
		node->next = tmp->next;
		free_data (tmp->data);
		free (tmp);
	}
}
