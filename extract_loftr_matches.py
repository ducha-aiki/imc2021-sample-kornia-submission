import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import torch
import kornia as K
import kornia.feature as KF
import h5py
import json
from PIL import Image
from adalam import AdalamFilter
from kornia_moons.feature import *
import pydegensac
from tqdm import tqdm
from PIL import Image
import argparse
from shutil import copyfile

def load_h5(filename):
    '''Loads dictionary from hdf5 file'''

    dict_to_load = {}
    try:
        with h5py.File(filename, 'r') as f:
            keys = [key for key in f.keys()]
            for key in keys:
                dict_to_load[key] = f[key][()]
    except:
        print('Cannot find file {}'.format(filename))
    return dict_to_load


def match_with_loftr(matcher, img1, img2, W, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    timg1_small = cv2.resize(img1, (W, H))
    coef_h1 = h1 / float(H)
    coef_w1 = w1 / float(W)
    timg2_small = cv2.resize(img2, (H, H))
    coef_h2 = h2 / float(H)
    coef_w2 = w2 / float(W)
    timg1 = K.image_to_tensor(timg1_small, None).float().to(device) / 255.
    timg2 = K.image_to_tensor(timg2_small, None).float().to(device) / 255.
    batch = {'image0': timg1, 'image1': timg2}
    with torch.no_grad():
        matcher(batch)
    src_pts = batch['mkpts0_f'].detach().cpu().numpy()
    dst_pts = batch['mkpts1_f'].detach().cpu().numpy()
    
    src_pts[:,0] *=coef_w1
    src_pts[:,1] *=coef_h1

    dst_pts[:,0] *=coef_w2
    dst_pts[:,1] *=coef_h2
    return src_pts, dst_pts


if __name__ == '__main__':
    parser = argparse.ArgumentParser("IMC2021 LoFTR sample submission script")
    parser.add_argument(
        "--resize_to_width", type=int, default=640, help='')
    parser.add_argument(
        "--resize_to_height", type=int, default=480, help='')
    parser.add_argument(
        "--data_path", type=str, default=os.path.join('..', 'imc-2021-data'))
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.path.join('extracted', 'loftr'),
        help='Path to store the features')
    W, H = args.resize_to_width, args.resize_to_height
    args = parser.parse_args()
    device = torch.device('cpu')
    device=torch.device('cuda')
    INPUT_DIR = args.data_path
    OUT_DIR = f'{args.save_path}-{W}x{H}'
    #datasets = [x for x in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, x))]
    #datasets = ['pragueparks']
    #datasets = ['pragueparks', 'phototourism', 'googleurban']
    datasets = ['phototourism']
    #datasets = ['googleurban']
    matcher = KF.LoFTR(pretrained='outdoor').eval().to(device)
    for ds in datasets:
        print (f"Processing dataset {ds}")
        ds_in_path = os.path.join(INPUT_DIR, ds)
        ds_out_path = os.path.join(OUT_DIR, ds)
        os.makedirs(ds_out_path, exist_ok=True)
        seqs = [x for x in os.listdir(ds_in_path) if os.path.isdir(os.path.join(ds_in_path,x))]
        for seq in seqs[::-1]:
            print (f"Processing sequence {seq}")
            if os.path.isdir(os.path.join(ds_in_path, seq, 'set_100')):
                seq_in_path = os.path.join(ds_in_path, seq, 'set_100', 'images')
            else:
                seq_in_path = os.path.join(ds_in_path, seq)
            seq_out_path = os.path.join(ds_out_path, seq)
            os.makedirs(seq_out_path, exist_ok=True)
            img_fnames = sorted(os.listdir(seq_in_path))[::-1]
            num_matches = []
            if os.path.isfile(f'{seq_out_path}/matches.h5'):
                print ('File exists, skipping')
                continue
            with h5py.File(f'{seq_out_path}/matches.h5', 'w') as f_m,\
                 h5py.File(f'{seq_out_path}/keypoints.h5', 'w') as f_kp:
                for i1, img1_fname in tqdm(enumerate(img_fnames)):
                    print (f'Matching image {img1_fname}')
                    img1_key = os.path.splitext(os.path.basename(img1_fname))[0]
                    img1_fname_full = os.path.join(seq_in_path, img1_fname)
                    img1 = cv2.imread(img1_fname_full, cv2.IMREAD_GRAYSCALE)
                    for img2_fname in tqdm(img_fnames[i1+1:]):
                        img2_key = os.path.splitext(os.path.basename(img2_fname))[0]
                        img2_fname_full = os.path.join(seq_in_path, img2_fname)
                        img2 = cv2.imread(img1_fname_full, cv2.IMREAD_GRAYSCALE)
                        match_key = f'{img1_key}-{img2_key}'
                        kp2_key = f'{img2_key}-{img1_key}'
                        src_pts, dst_pts = match_with_loftr(matcher, img1, img2, W, H)
                        idxs = np.stack([np.arange(len(src_pts)), np.arange(len(src_pts))],axis=-1)
                        num_matches.append(len(idxs))
                        if len(idxs) == 0:
                            idxs = np.empty([0, 2], dtype=np.int32)
                        idxs = idxs.T
                        assert idxs.shape[0] == 2
                        f_m[match_key] = idxs
                        f_kp[match_key] = src_pts
                        f_kp[kp2_key] = dst_pts
            print(f'Finished processing "{ds}/{seq}" -> {np.array(num_matches).mean()} matches/image')
