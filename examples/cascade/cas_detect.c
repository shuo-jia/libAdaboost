#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cascade.h"
#include "image.h"

// 文件名最大长度
#define MAX_FILENAME 128
// 图片所在目录
#define DIRPATH "./BioID-FaceDatabase-V1.2/"
// 训练集数量，将跳过训练集
#define TRAIN_COUNT 1000
// 标注文件
#define MARKPATH "./p_mark"
// 模型保存路径
#define MODEL_PATH "./cascade_data"

static void rect_print(struct cascade *pcascade, const char *fname,
		       const struct haar_ada_handles *handles);

int main (void)
{
	FILE *model_file;
	if ((model_file = fopen(MODEL_PATH, "rb")) == NULL) {
		fprintf (stderr, "Can't open file \"%s\".\n", MODEL_PATH);
		exit(EXIT_FAILURE);
	}

	struct cascade cascade;
	struct haar_ada_handles handles;
	ada_set_haar (&handles, ADA_ASYM_IMP, ADA_GA);
	cas_read (&cascade, model_file, &handles);
	rect_print (&cascade, MARKPATH, &handles);

	cas_free (&cascade, &handles);
	fclose (model_file);
	return 0;
}

void rect_print(struct cascade *pcascade, const char *fname,
		const struct haar_ada_handles *handles)
{
	char filename[MAX_FILENAME];
	strcpy(filename, DIRPATH);
	char *ptr_file = filename + strlen(filename);
	FILE *mark = fopen(fname, "r");
	if (mark == NULL) {
		perror("Can't open file.\n");
		exit(EXIT_FAILURE);
	}

	struct image *img;
	struct cas_det_rect *prect;
	link_iter iter;
	struct link_list rect_list;
	// 跳过训练集部分
	for (int i = 0; i < TRAIN_COUNT; ++i)
		fscanf(mark, "%*s %*d %*d %*d %*d");
	// 在测试集上进行检测
	while (fscanf(mark, "%s %*d %*d %*d %*d", ptr_file) == 1) {
		if ((img = imread_pgm(filename)) == NULL) {
			perror("Can't open image.\n");
			fclose(mark);
			exit(EXIT_FAILURE);
		}
		rect_list = cas_detect(pcascade, img->height, img->width,
				       (void *)img->img, 3, handles);
		iter = link_list_start_iter(&rect_list);
		printf("file: %s\n", ptr_file);
		while (link_list_check_end(iter)) {
			prect = link_list_get_data(iter);
			printf("x: %4d \t y: %4d \t len: %4d\n",
			       prect->rect.start_x, prect->rect.start_y,
			       prect->rect.len);
			link_list_next_iter(&iter);
		}
		free(img);
		link_list_free_full(&rect_list, free);
	}
	fclose(mark);
}
