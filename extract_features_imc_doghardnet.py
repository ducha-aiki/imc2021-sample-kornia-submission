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


def extract_features(img_fname, detector, descriptor, device, visualize=False):
    img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
    if visualize:
        plt.imshow(img)
    kpts = detector.detect(img, None)[:8000]
    # We will not train anything, so let's save time and memory by no_grad()
    with torch.no_grad():
        timg = K.image_to_tensor(img, False).float()/255.
        timg = timg.to(device)
        timg_gray = K.color.rgb_to_grayscale(timg)
        # kornia expects keypoints in the local affine frame format. 
        # Luckily, kornia_moons has a conversion function
        lafs = laf_from_opencv_SIFT_kpts(kpts, device=device)
        patches = KF.extract_patches_from_pyramid(timg_gray, lafs, 32)
        B, N, CH, H, W = patches.size()
        # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
        # So we need to reshape a bit :) 
        descs = descriptor(patches.view(B * N, CH, H, W)).view(B * N, -1).detach().cpu().numpy()    
    return kpts, descs, lafs.reshape(B*N, 2, 3).detach().cpu().numpy()

def convert_kpts_to_imc(cv2_kpts):
    keypoints = np.array([(x.pt[0], x.pt[1]) for x in cv2_kpts ]).reshape(-1, 2)
    scales = np.array([x.size for x in cv2_kpts ]).reshape(-1, 1)
    angles = np.array([x.angle for x in cv2_kpts ]).reshape(-1, 1)
    responses = np.array([x.response for x in cv2_kpts]).reshape(-1, 1)
    return keypoints, scales, angles, responses

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
    sift_det =  cv2.SIFT_create(args.num_keypoints, contrastThreshold=-10000, edgeThreshold=-10000)
    # HardNet descriptor
    hardnet = KF.HardNet(True).eval().to(device)
    # Affine shape estimator
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
                 h5py.File(f'{seq_out_path}/descriptors.h5', mode='w') as f_desc, \
                 h5py.File(f'{seq_out_path}/scores.h5', mode='w') as f_score, \
                 h5py.File(f'{seq_out_path}/lafs.h5', mode='w') as f_laf, \
                 h5py.File(f'{seq_out_path}/angles.h5', mode='w') as f_ang, \
                 h5py.File(f'{seq_out_path}/scales.h5', mode='w') as f_scale:
                for img_fname in tqdm(img_fnames):
                    img_fname_full = os.path.join(seq_in_path, img_fname)
                    key = os.path.splitext(os.path.basename(img_fname))[0]
                    kpts, descs, lafs = extract_features(img_fname_full,  sift_det,  hardnet, device, False)
                    keypoints, scales, angles, responses = convert_kpts_to_imc(kpts)
                    f_kp[key] = keypoints[:8000]
                    f_laf[key] = lafs[:8000]
                    f_desc[key] = descs.reshape(-1, 128)[:8000]
                    f_score[key] = responses[:8000]
                    f_ang[key] = angles[:8000]
                    f_scale[key] = scales[:8000]
                    num_kp.append(len(keypoints))
                print(f'Finished processing "{ds}/{seq}" -> {np.array(num_kp).mean()} features/image')

