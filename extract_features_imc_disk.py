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


def extract_features(img_fname, dsk, device, nkpts = 2048, visualize=False):
    img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
    # We will not train anything, so let's save time and memory by no_grad()
    with torch.no_grad():
        timg = K.image_to_tensor(img, False).float()/255.
        timg = timg.to(device)
        features1 = dsk(timg, nkpts, pad_if_not_divisible=True)[0]
        kps1, descs1 = features1.keypoints, features1.descriptors
    return kps1, descs1


if __name__ == '__main__':
    parser = argparse.ArgumentParser("IMC2021 sample submission script")
    parser.add_argument(
        "--num_keypoints", type=int, default=8000, help='Number of keypoints')
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
    disk = KF.DISK.from_pretrained("depth").to(device)
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
                    keypoints, descs = extract_features(img_fname_full,  disk, device,args.num_keypoints,  False)
                    f_kp[key] = keypoints[:8000].reshape(-1, 2).detach().cpu().numpy()
                    f_desc[key] = descs.reshape(-1,128).detach().cpu().numpy()
                    num_kp.append(len(keypoints))
                print(f'Finished processing "{ds}/{seq}" -> {np.array(num_kp).mean()} features/image')

