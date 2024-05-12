# TODO:Crop big size images
import os
import cv2
import tifffile as tiff


def make_dataset(count, im, base_num, img_type, tif, overlap):
    h, w = im.shape[:2]
    stride = int(base_num * (1 - overlap))
    if h < base_num or w < base_num:
        print('Original image size is too small!')
        quit()

    else:
        if (h - base_num) % stride == 0 and (w - base_num) % stride != 0:
            for i in range((h - base_num) // stride + 1):
                for j in range((w - base_num) // stride + 2):
                    if j == (w - base_num) // stride + 1 and i < (h - base_num) // stride + 1:
                        if len(im.shape) == 2:
                            new_img = im[stride * i:stride * i + base_num:, -base_num:]
                        else:
                            new_img = im[stride * i:stride * i + base_num:, -base_num:, :]
                    else:
                        if len(im.shape) == 2:
                            new_img = im[stride * i:stride * i + base_num, stride * j:stride * j + base_num]
                        else:
                            new_img = im[stride * i:stride * i + base_num, stride * j:stride * j + base_num, :]

                    if tif:
                        if img_type == 'tif':
                            tiff.imwrite(save_imgs_path + str(count).zfill(5) + ".tif", new_img)
                        else:
                            if len(im.shape) != 2:
                                b, g, r = cv2.split(new_img)
                                new_img = cv2.merge([r, g, b])
                            cv2.imwrite(save_imgs_path + str(count).zfill(5) + ".{}".format(img_type), new_img)
                    else:
                        cv2.imwrite(save_imgs_path + str(count).zfill(5) + ".{}".format(img_type), new_img)
                    count += 1

        elif (w - base_num) % stride == 0 and (h - base_num) % stride != 0:
            for i in range((h - base_num) // stride + 2):
                for j in range((w - base_num) // stride + 1):
                    if i == (h - base_num) // stride + 1 and j < (w - base_num) // stride + 1:
                        if len(im.shape) == 2:
                            new_img = im[-base_num:, stride * j:stride * j + base_num]
                        else:
                            new_img = im[-base_num:, stride * j:stride * j + base_num, :]
                    else:
                        if len(im.shape) == 2:
                            new_img = im[stride * i:stride * i + base_num, stride * j:stride * j + base_num]
                        else:
                            new_img = im[stride * i:stride * i + base_num, stride * j:stride * j + base_num, :]

                    if tif:
                        if img_type == 'tif':
                            tiff.imwrite(save_imgs_path + str(count).zfill(5) + ".tif", new_img)
                        else:
                            if len(im.shape) != 2:
                                b, g, r = cv2.split(new_img)
                                new_img = cv2.merge([r, g, b])
                            cv2.imwrite(save_imgs_path + str(count).zfill(5) + ".{}".format(img_type), new_img)
                    else:
                        cv2.imwrite(save_imgs_path + str(count).zfill(5) + ".{}".format(img_type), new_img)
                    count += 1

        elif (w - base_num) % stride == 0 and (h - base_num) % stride == 0:
            for i in range((h - base_num) // stride + 1):
                for j in range((w - base_num) // stride + 1):
                    if len(im.shape) == 2:
                        new_img = im[stride * i:stride * i + base_num, stride * j:stride * j + base_num]
                    else:
                        new_img = im[stride * i:stride * i + base_num, stride * j:stride * j + base_num, :]

                    if tif:
                        if img_type == 'tif':
                            tiff.imwrite(save_imgs_path + str(count).zfill(5) + ".tif", new_img)
                        else:
                            if len(im.shape) != 2:
                                b, g, r = cv2.split(new_img)
                                new_img = cv2.merge([r, g, b])
                            cv2.imwrite(save_imgs_path + str(count).zfill(5) + ".{}".format(img_type), new_img)
                    else:
                        cv2.imwrite(save_imgs_path + str(count).zfill(5) + ".{}".format(img_type), new_img)
                    count += 1

        else:
            for i in range((h - base_num) // stride + 2):
                for j in range((w - base_num) // stride + 2):
                    if i == (h - base_num) // stride + 1 and j < (w - base_num) // stride + 1:
                        if len(im.shape) == 2:
                            new_img = im[-base_num:, stride * j:stride * j + base_num]
                        else:
                            new_img = im[-base_num:, stride * j:stride * j + base_num, :]
                    elif j == (w - base_num) // stride + 1 and i < (h - base_num) // stride + 1:
                        if len(im.shape) == 2:
                            new_img = im[stride * i:stride * i + base_num:, -base_num:]
                        else:
                            new_img = im[stride * i:stride * i + base_num:, -base_num:, :]
                    elif i == (h - base_num) // stride + 1 and j == (w - base_num) // stride + 1:
                        if len(im.shape) == 2:
                            new_img = im[-base_num:, -base_num:]
                        else:
                            new_img = im[-base_num:, -base_num:, :]
                    else:
                        if len(im.shape) == 2:
                            new_img = im[stride * i:stride * i + base_num, stride * j:stride * j + base_num]
                        else:
                            new_img = im[stride * i:stride * i + base_num, stride * j:stride * j + base_num, :]

                    if tif:
                        if img_type == 'tif':
                            tiff.imwrite(save_imgs_path + str(count).zfill(5) + ".tif", new_img)
                        else:
                            if len(im.shape) != 2:
                                b, g, r = cv2.split(new_img)
                                new_img = cv2.merge([r, g, b])
                            cv2.imwrite(save_imgs_path + str(count).zfill(5) + ".{}".format(img_type), new_img)
                    else:
                        cv2.imwrite(save_imgs_path + str(count).zfill(5) + ".{}".format(img_type), new_img)
                    count += 1


# args
imgs_path = r""  # dir containing large size images to be cropped
save_imgs_path = r""  # dir to save cropped images
is_graylabel = False  # whether the path is grayscale label images
size = 512  # cropped size
img_type = 'tif'  # the format in which the images are saved
overlap = 0  # the overlap ratio of crop patches

imgs_dir = os.listdir(imgs_path)
if len(os.listdir(save_imgs_path)) == 0:
    count = 0
else:
    count = int(os.listdir(save_imgs_path)[-1].split(".")[0]) + 1
begin_num = count
for i, j in enumerate(imgs_dir):
    name = j.split(".")[1]
    if name == 'tif':
        im = tiff.imread(imgs_path + j)
        tif = True
    else:
        im = cv2.imread(imgs_path + j, -1) if is_graylabel else cv2.imread(imgs_path + j)
        tif = False
    img_type = 'png' if is_graylabel else img_type  # prevent labels being garbled after cropping
    make_dataset(count, im, size, img_type, tif, overlap)
    count = len(os.listdir(save_imgs_path))
    print('No.{} image is cropped !'.format(i + 1))
print('All images have been cropped, the total of {} small size images have been obtained'.format(count - begin_num))
