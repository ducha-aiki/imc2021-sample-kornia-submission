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

def opencv_from_imc(kps, sizes, angles):
    return [cv2.KeyPoint(kp[0], kp[1], float(s), float(a)) for  kp, s, a in zip(kps, sizes, angles)]

def get_data(kps, angles, scales, descs, img_key):
    kp1 = kps[img_key]
    s1 = scales[img_key]
    a1 = angles[img_key]
    descs1 = descs[img_key]
    return kp1, s1, a1, descs1

def match_adalam_with_degensac(kp1, kp2, s1, s2, a1, a2, descs1, descs2,
                               h1, w1, h2, w2, ds_name='phototourism'):
    matcher = AdalamFilter()
    
    idxs = matcher.match_and_filter(kp1, kp2,
                            descs1, descs2,
                            im1shape=(h1,w1), 
                            im2shape=(h2,w2),
                            o1=a1.reshape(-1),
                            o2=a2.reshape(-1),
                            s1=s1.reshape(-1),
                            s2=s2.reshape(-1)).detach().cpu().numpy()
    if len(idxs) <= 8:
        return np.empty((0,1), dtype=np.float32), np.empty((0,2), dtype=np.int32)
    src_pts = kp1[idxs[:,0]]
    dst_pts = kp2[idxs[:,1]]
    
    max_iters = 100000
    if ds_name.lower() == 'phototourism':
        inl_th = 0.5
    elif ds_name.lower() == 'pragueparks':
        inl_th = 1.5 
    elif ds_name.lower() == 'googleurban':
        inl_th = 0.75
    else:
        raise ValueError('Unknown dataset')
    F, inliers_mask = pydegensac.findFundamentalMatrix(src_pts, dst_pts, inl_th, 0.999999, max_iters)
    out_idxs = idxs[inliers_mask]
    # AdaLAM does not provide confidence score, so we will create dummy one
    dists = np.ones_like(out_idxs)[:,0] 
    return dists, out_idxs

if __name__ == '__main__':
    parser = argparse.ArgumentParser("IMC2021 sample submission script")
    parser.add_argument(
        "--num_keypoints", type=int, default=8000, help='Number of keypoints')
    parser.add_argument(
        "--data_path", type=str, default=os.path.join('..', 'imc-2021-data'))
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.path.join('extracted', 'cv2-dog-affnet-hardnet8'),
        help='Path to store the features')
    args = parser.parse_args()
    device = torch.device('cpu')
    INPUT_DIR = args.data_path
    OUT_DIR = args.save_path
    datasets = [x for x in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, x))]
    #datasets = ['pragueparks']
    #datasets = ['pragueparks', 'phototourism', 'googleurban']
    #datasets = ['phototourism']
    #datasets = ['googleurban']
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
            kps = load_h5(os.path.join(seq_out_path, 'keypoints.h5'))
            angles = load_h5(os.path.join(seq_out_path, 'angles.h5'))
            scales = load_h5(os.path.join(seq_out_path, 'scales.h5'))
            descs = load_h5(os.path.join(seq_out_path, 'descriptors.h5'))
            img_fnames = sorted(os.listdir(seq_in_path))[::-1]
            num_matches = []
            if os.path.isfile(f'{seq_out_path}/matches_stereo_0.h5') and not os.path.isfile(f'{seq_out_path}/matches-stereo.h5'):
                print ('File exists, skipping')
                continue
            with h5py.File(f'{seq_out_path}/matches_stereo_0.h5', 'w') as f_m:
                for i1, img1_fname in tqdm(enumerate(img_fnames)):
                    print (f'Matching image {img1_fname}')
                    img1_key = os.path.splitext(os.path.basename(img1_fname))[0]
                    img1_fname_full = os.path.join(seq_in_path, img1_fname)
                    img1 = Image.open(img1_fname_full)
                    w1, h1 = img1.size
                    kp1, s1, a1, descs1 = get_data(kps, angles, scales, descs, img1_key)
                    for img2_fname in tqdm(img_fnames[i1+1:]):
                        img2_key = os.path.splitext(os.path.basename(img2_fname))[0]
                        img2_fname_full = os.path.join(seq_in_path, img2_fname)
                        img2 = Image.open(img2_fname_full)
                        w2, h2 = img2.size
                        match_key = f'{img1_key}-{img2_key}'
                        kp2, s2, a2, descs2 = get_data(kps, angles, scales, descs, img2_key)
                        _, idxs = match_adalam_with_degensac(kp1, kp2, s1, s2, a1, a2, descs1, descs2,
                                       h1, w1, h2, w2, ds_name=ds)
                        num_matches.append(len(idxs))
                        if len(idxs) == 0:
                            idxs = np.empty([0, 2], dtype=np.int32)
                        idxs = idxs.T
                        assert idxs.shape[0] == 2
                        f_m[match_key] = idxs
            print(f'Finished processing "{ds}/{seq}" -> {np.array(num_matches).mean()} matches/image')
            copyfile(f'{seq_out_path}/matches_stereo_0.h5', f'{seq_out_path}/matches_multiview.h5')
