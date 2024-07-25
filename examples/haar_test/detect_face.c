/* 用矩形框出图片中的人脸，按 q 退出 */

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include "cascade.h"
#include "image.h"
#include "fb_image.h"

#define MODEL_PATH "./cascade_data"
#define DIRPATH "../BioID-FaceDatabase-V1.2/"
// 训练集数量，将跳过训练集
#define TRAIN_COUNT 1
// 文件名最大长度
#define MAX_FILENAME 128

static void rect_cast (struct rectangle *rect, const struct cas_rect * cas_rect);
static void show_face (__Canvas *pcanvas, struct cascade *pcascade,
		const struct haar_ada_handles * handles, DIR *dir);

int main (void)
{
	DIR *dir;
	FILE *model_file;
	if ((dir = opendir (DIRPATH)) == NULL) {
		fprintf (stderr, "Can't open dir \"%s\".\n", DIRPATH);
		return 1;
	}
	if ((model_file = fopen(MODEL_PATH, "rb")) == NULL) {
		fprintf (stderr, "Can't open file \"%s\".\n", MODEL_PATH);
		closedir (dir);
		return 1;
	}

	__Screen screen;
	__Canvas canvas;
	InitScreen (&screen, "/dev/fb0");
	__Position start = {0, 0};
	SetCanvas (&screen, &canvas, &start, screen.height, screen.width);

	struct cascade cascade;
	struct haar_ada_handles handles;
	ada_set_haar (&handles, ADA_ASYM, ADA_GA);
	cas_read (&cascade, model_file, &handles);
	show_face (&canvas, &cascade, &handles, dir);

	cas_free (&cascade, &handles);
	FreeCanvas (&canvas);
	CloseScreen (&screen);
	fclose (model_file);
	closedir (dir);
	return 0;
}

void show_face (__Canvas *pcanvas, struct cascade *pcascade,
		const struct haar_ada_handles * handles, DIR *dir)
{
	char filename[MAX_FILENAME];
	strcpy (filename, DIRPATH);
	char *ptr_file = filename + strlen(filename);
	char *ptr_dot;

	struct image *img;
	struct cas_det_rect * prect;
	struct rectangle rect;

	char ch;
	int count = 0;
	struct dirent *ent;
	struct link_list rect_list;
	link_iter iter;
	while ((ent = readdir (dir)) != NULL) {
		if (! (ptr_dot = strrchr(ent->d_name, '.')) ||
			strcmp (ptr_dot, ".pgm"))
			continue;
		++count;
		if (count < TRAIN_COUNT)
			continue;

		strcpy(ptr_file, ent->d_name);
		img = imread_pgm (filename);
		imshow (pcanvas, img);
		rect_list = cas_detect (pcascade, img->height, img->width,
				(void *)img->img, 3, handles);
		iter = link_list_start_iter(&rect_list);
		while (link_list_check_end (iter)) {
			prect = link_list_get_data (iter);
			rect_cast (&rect, &prect->rect);
			show_rect (pcanvas, NULL, &rect, 0xFFFF);
			link_list_next_iter (&iter);
		}

		free (img);
		link_list_free_full (&rect_list, free);
		if ((ch = getchar()) == 'q')
			break;
	}
}

void rect_cast (struct rectangle *rect, const struct cas_rect * cas_rect)
{
	rect->start_x = cas_rect->start_x;
	rect->start_y = cas_rect->start_y;
	rect->width = rect->height = cas_rect->len;
}
