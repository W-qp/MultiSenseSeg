# TODO:Predict and visualize
import argparse
import glob
import logging
import os
import shutil
import time
import warnings
import cv2
import gdal
import numpy as np
import tifffile as tiff
import torch
import torch.nn.functional as F
from PIL import Image
import json
from model.build_model.Build_MultiSenseSeg import Build_MultiSenseSeg
from utils.dataset import BasicDataset
from utils.logger import logger

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None
start_time = time.time()


def predict_img(net, full_img, device):
    global size
    net.eval()
    for i in range(len(full_img)):
        size = full_img[i].shape[:2]
        img = BasicDataset.preprocess(full_img[i])
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)
        full_img[i] = img / 255  # normalized

    with torch.no_grad():
        pred = net(full_img)
        if net.n_classes > 1:
            pred = F.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
        else:
            pred = (pred > 0.5).int().squeeze(dim=0)

        pred = pred.cpu().squeeze().numpy().astype(np.uint8)
        pred = cv2.resize(pred, size)

    return pred


def mask_to_image(mask):
    if net.n_classes > 2:
        assert os.path.exists(json_dict['classes']), f"cannot find {json_dict['classes']} file!"
        classes = json.load(open(json_dict['classes'], 'r'))
        image = np.zeros([mask.shape[0], mask.shape[1], 3])
        for i, v in enumerate(classes.values()):
            idx = np.where(mask == i)
            image[idx] = v

        image = Image.fromarray(image.astype(np.uint8))

    else:
        image = Image.fromarray((mask * 255).astype(np.uint8))

    return image


def create_name(name):
    ij = name.split("/")[-1].split(".")[0].split("_")
    i, j = int(ij[-2]), int(ij[-1])
    return i, j


def crop_image(idx, b, save_path):
    add_doc(json_dict['crop'] + f"crop{idx}/")
    doc = os.path.splitext(os.listdir(json_dict[f'{idx}_input_img'])[0])[-1]
    img_path = json_dict[f'{idx}_input_img'] + big_img_name + doc
    if doc == '.tif':
        big_img = tiff.imread(img_path)
    else:
        big_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        if len(big_img.shape) == 3:
            big_img = cv2.cvtColor(big_img, cv2.COLOR_BGR2RGB)

    h, w = big_img.shape[:-1] if len(big_img.shape) == 3 else big_img.shape
    if h < b or w < b:
        print('Original image size is too small!')
        quit()
    else:
        if h % b == 0 and w % b != 0:
            for i in range(h // b):
                for j in range(w // b + 1):
                    if j == w // b and i < h // b:
                        if len(big_img.shape) == 2:
                            new_img = big_img[b * i:b * i + b:, -b:]
                        else:
                            new_img = big_img[b * i:b * i + b:, -b:, :]
                    else:
                        if len(big_img.shape) == 2:
                            new_img = big_img[b * i:b * i + b, b * j:b * j + b]
                        else:
                            new_img = big_img[b * i:b * i + b, b * j:b * j + b, :]

                    tiff.imwrite(save_path + "_{}_{}.tif".format(i, j), new_img)

        elif w % b == 0 and h % b != 0:
            for i in range(h // b + 1):
                for j in range(w // b):
                    if i == h // b and j < w // b:
                        if len(big_img.shape) == 2:
                            new_img = big_img[-b:, b * j:b * j + b]
                        else:
                            new_img = big_img[-b:, b * j:b * j + b, :]
                    else:
                        if len(big_img.shape) == 2:
                            new_img = big_img[b * i:b * i + b, b * j:b * j + b]
                        else:
                            new_img = big_img[b * i:b * i + b, b * j:b * j + b, :]

                    tiff.imwrite(save_path + "_{}_{}.tif".format(i, j), new_img)

        elif w % b == 0 and h % b == 0:
            for i in range(h // b):
                for j in range(w // b):
                    if len(big_img.shape) == 2:
                        new_img = big_img[b * i:b * i + b, b * j:b * j + b]
                    else:
                        new_img = big_img[b * i:b * i + b, b * j:b * j + b, :]

                    tiff.imwrite(save_path + "_{}_{}.tif".format(i, j), new_img)

        else:
            for i in range(h // b + 1):
                for j in range(w // b + 1):
                    if i == h // b and j < w // b:
                        if len(big_img.shape) == 2:
                            new_img = big_img[-b:, b * j:b * j + b]
                        else:
                            new_img = big_img[-b:, b * j:b * j + b, :]
                    elif j == w // b and i < h // b:
                        if len(big_img.shape) == 2:
                            new_img = big_img[b * i:b * i + b:, -b:]
                        else:
                            new_img = big_img[b * i:b * i + b:, -b:, :]
                    elif i == h // b and j == w // b:
                        if len(big_img.shape) == 2:
                            new_img = big_img[-b:, -b:]
                        else:
                            new_img = big_img[-b:, -b:, :]
                    else:
                        if len(big_img.shape) == 2:
                            new_img = big_img[b * i:b * i + b, b * j:b * j + b]
                        else:
                            new_img = big_img[b * i:b * i + b, b * j:b * j + b, :]

                    tiff.imwrite(save_path + "_{}_{}.tif".format(i, j), new_img)


def concatenation(b):
    if os.path.splitext(big_img_dir)[1] == '.tif':
        h, w = tiff.imread(json_dict['1_input_img'] + big_img_dir).shape[:2]
    else:
        h, w = cv2.imread(json_dict['1_input_img'] + big_img_dir).shape[:2]
    input_preds = glob.glob(json_dict['crop'] + "pred_imgs/*")

    if net.n_classes > 2:
        rs = np.zeros((h, w, 3))
        for n in input_preds:
            i, j = create_name(n)
            photo = cv2.imread(n)
            logging.info(f'Concatenating image with index i={i},j={j} ...\n')

            if i == h // b and j < w // b:
                rs[-b:, b * j:b * j + b, :] = photo
            elif j == w // b and i < h // b:
                rs[b * i:b * i + b:, -b:, :] = photo
            elif i == h // b and j == w // b:
                rs[-b:, -b:, :] = photo
            else:
                rs[b * i:b * i + b, b * j:b * j + b, :] = photo

    else:
        rs = np.zeros((h, w))
        for n in input_preds:
            i, j = create_name(n)
            photo = cv2.imread(n, 0)
            logging.info(f'Concatenating image with index i={i},j={j} ...\n')

            if i == h // b and j < w // b:
                rs[-b:, b * j:b * j + b] = photo
            elif j == w // b and i < h // b:
                rs[b * i:b * i + b:, -b:] = photo
            elif i == h // b and j == w // b:
                rs[-b:, -b:] = photo
            else:
                rs[b * i:b * i + b, b * j:b * j + b] = photo

    return rs


def create_tif(array, tif_dir, resultTif):
    ori_tif = gdal.Open(tif_dir)
    trans = ori_tif.GetGeoTransform()
    proj = ori_tif.GetProjection()
    width = ori_tif.RasterXSize
    height = ori_tif.RasterYSize
    driver = gdal.GetDriverByName("GTiff")
    if len(array.shape) == 2:
        new_tif = driver.Create(resultTif, width, height, 1, gdal.GDT_Int16)
        new_tif.GetRasterBand(1).WriteArray(array)
    else:
        new_tif = driver.Create(resultTif, width, height, array.shape[-1])
        for i in range(array.shape[-1]):
            new_tif.GetRasterBand(i + 1).WriteArray(array[..., i])
    new_tif.SetProjection(proj)
    new_tif.SetGeoTransform(trans)

    del new_tif


def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)


def add_doc(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        del_files(dir)


def get_args():
    parser = argparse.ArgumentParser(description='Predict', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_chans', type=int and tuple, default=(3, 3), help='Channels of input images')
    parser.add_argument('--json_path', type=str, default=r".../predict.json", help='.json path')
    parser.add_argument('--base_num', type=int, default=512, help='Crop to the size of the predicted images')
    parser.add_argument('--gpu_id', type=int, default=0, help='Number of gpu')
    parser.add_argument('--net_type', type=str, default='MultiSenseSeg', help='Net type')
    parser.add_argument('--if_save_tif', default=False, help='Whether to save the prediction .tif result')
    parser.add_argument('--scale', type=float, default=1, help='Scale factor for the input images')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    in_chans = args.in_chans
    n_img = len(in_chans) if isinstance(in_chans, tuple) else 1
    json_path = args.json_path
    base_num = args.base_num
    gpu_id = args.gpu_id
    net_type = args.net_type
    save_tif = args.if_save_tif
    scale = args.scale
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    assert os.path.exists(json_path), f"cannot find {json_path} file!"
    json_dict = json.load(open(json_path, 'r'))
    n_classes = len(json.load(open(json_dict['classes'], 'r')))
    img_size = int(base_num * scale)

    if net_type == 'MultiSenseSeg':
        net = Build_MultiSenseSeg(n_classes=n_classes, in_chans=in_chans)
    else:
        raise NotImplementedError(f"net type:'{net_type}' does not exist, please check the 'net_type' arg!")

    logger(net_type, json_dict['runs'])
    logging.info('Loading model {}'.format(json_dict['ckpt'] + f"{net_type}.pth"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(json_dict['ckpt'] + f"{net_type}.pth", map_location=device))
    logging.info('Sliding window clipping...\n')

    add_doc(json_dict['crop'])
    add_doc(json_dict['crop'] + "pred_imgs/")
    for big_img_dir in os.listdir(json_dict['1_input_img']):
        img_time = time.time()
        big_img_name = os.path.splitext(big_img_dir)[0]
        for i in range(1, n_img+1):
            crop_image(i, base_num, json_dict['crop'] + f"crop{i}/crop")

        for i in range(len(os.listdir(json_dict['crop'] + f"crop1/"))):
            in_imgs = []
            for j in range(1, n_img+1):
                img_dir = glob.glob(json_dict['crop'] + f"crop{j}/*")[i]
                in_imgs.append(tiff.imread(img_dir))

            img_name = os.path.splitext(glob.glob(json_dict['crop'] + "crop1/*")[i])[0]
            logging.info(f'Predicting image {img_name} ...\n')
            mask = predict_img(net, in_imgs, device)
            pred_img = mask_to_image(mask)
            idx_h, idx_w = create_name(img_name)
            pred_img.save(json_dict['crop'] + f"pred_imgs/pred_{idx_h}_{idx_w}.png")

        big_pred_img = concatenation(base_num)
        cv2.imwrite(json_dict['save_png'] + f"{big_img_name}.png", big_pred_img)

        if save_tif:
            tif_img = cv2.cvtColor(big_pred_img.astype(np.uint8), cv2.COLOR_BGR2RGB) if net.n_classes > 2 else big_pred_img
            create_tif(tif_img, json_dict['add_geo'] + f"{big_img_name}.tif", json_dict['save_tif'] + f"{big_img_name}.tif")

        logging.info(f"The current image took {round((time.time() - img_time) / 3600, 4)} hours to complete the prediction!\n")
        del big_pred_img
        for num in range(1, n_img + 1):
            del_files(json_dict['crop'] + f"crop{num}")
        del_files(json_dict['crop'] + "pred_imgs/")

    shutil.rmtree(json_dict['crop'])
    end_time = time.time()
    run_time = end_time - start_time
    logging.info('The entire {} model prediction time is: {} hours.'.format(net_type, round(run_time / 3600, 3)))
