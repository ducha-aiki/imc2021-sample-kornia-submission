import os
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *
import argparse
from lightglue import SuperPoint, ALIKED
from lightglue.utils import load_image, rbd


def extract_features(img_fname, dsk, device, nkpts = 2048, visualize=False):
    img = load_image(img_fname).to(device)
    with torch.no_grad():
        feats0 = dsk.extract(img)
        kps1 = feats0['keypoints'][0]
        descs1 = feats0['descriptors'][0]
    return kps1, descs1


if __name__ == '__main__':
    parser = argparse.ArgumentParser("IMC2021 sample submission script")
    parser.add_argument(
        "--num_keypoints", type=int, default=8000, help='Number of keypoints')
    parser.add_argument(
        "--feature", type=str, default='superpoint')
    parser.add_argument(
        "--data_path", type=str, default=os.path.join('..', 'imc-2021-data'))
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.path.join('extracted', 'cv2-dog-hardnet'),
        help='Path to store the features')
    args = parser.parse_args()
    device = torch.device('cpu')
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print ("GPU mode")
    except:
        print ('CPU mode')
    # SIFT (DoG) Detector
    if args.feature == 'superpoint':
        extractor = SuperPoint(max_num_keypoints=args.num_keypoints).eval().to(device)
    else:
        extractor = ALIKED(max_num_keypoints=args.num_keypoints, detection_threshold=0.0001).eval().to(device)
    INPUT_DIR = args.data_path
    OUT_DIR = args.save_path
    os.makedirs(OUT_DIR, exist_ok=True)
    datasets = [x for x in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, x))]
    for ds in datasets:
        ds_in_path = os.path.join(INPUT_DIR, ds)
        ds_out_path = os.path.join(OUT_DIR, ds)
        os.makedirs(ds_out_path, exist_ok=True)
        seqs = [x for x in os.listdir(ds_in_path) if os.path.isdir(os.path.join(ds_in_path, x))]
        for seq in seqs:
            if os.path.isdir(os.path.join(ds_in_path, seq, 'set_100')):
                seq_in_path = os.path.join(ds_in_path, seq, 'set_100', 'images')
            else:
                seq_in_path = os.path.join(ds_in_path, seq)
            seq_out_path = os.path.join(ds_out_path, seq)
            os.makedirs(seq_out_path, exist_ok=True)
            img_fnames = os.listdir(seq_in_path)
            num_kp = []
            with h5py.File(f'{seq_out_path}/keypoints.h5', mode='w') as f_kp, \
                    h5py.File(f'{seq_out_path}/descriptors.h5', mode='w') as f_desc:
                for img_fname in tqdm(img_fnames):
                    img_fname_full = os.path.join(seq_in_path, img_fname)
                    key = os.path.splitext(os.path.basename(img_fname))[0]
                    keypoints, descs = extract_features(img_fname_full,  extractor, device,args.num_keypoints,  False)
                    f_kp[key] = keypoints.reshape(-1, 2).detach().cpu().numpy()
                    f_desc[key] = descs.detach().cpu().numpy()
                    num_kp.append(len(keypoints))
                print(f'Finished processing "{ds}/{seq}" -> {np.array(num_kp).mean()} features/image')

