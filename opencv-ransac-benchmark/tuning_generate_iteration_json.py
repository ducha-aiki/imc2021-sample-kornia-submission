import os
import numpy as np
import json

inl_ths = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]

if __name__ == '__main__':
    hashname='tuning-'
    base_res_dict = {}
    datasets = ['phototourism', 'pragueparks', 'googleurban']
    for dset in datasets:
        base_res_dict[dset] = {}

    task = 'stereo'
    metric = 'qt_auc_10_th_0.1'
    ransacs = ['cv2-usacdef-f',
               'cv2-usacmagsac-f',
               'cv2-usacfast-f',
               'cv2-usacaccurate-f', 'cv2-ransac-f']
    from create_cv2_ransac_configs import *
    for R in ransacs:
        res_dict = deepcopy(base_res_dict)
        for inl_th in inl_ths:
            res_fname = os.path.join('../image-matching-benchmark/packed-val', f'{hashname}-{R}-inlth-{inl_th}.json')
            try:
                with open(res_fname, 'r') as f:
                    results = json.load(f)
            except:
                 print (f'No results for {R} {inl_th}')
                 continue
            submission_name = results['config']['metadata']['method_name']
            res_dict[inl_th] = {}
            for dset in datasets:
                mAA = results[dset]['results']['allseq'][task]['run_avg'][metric]['mean']
                res_dict[dset][inl_th] = mAA
        final_ths = {}
        for i, dset in enumerate(datasets):
            inl_ths = []
            mAAs = []
            for inl_th, mAA in res_dict[dset].items():
                inl_ths.append(inl_th)
                mAAs.append(mAA)
            best_th_idx = np.argmax(np.array(mAAs))
            best_th = inl_ths[best_th_idx]
            best_mAA = mAAs[best_th_idx]
            print (mAAs)
            print (f'Best {R} {dset} mAA = {best_mAA:.4f} with inl_th = {best_th}')
            final_ths[dset] = best_th
        configs = []
        current_config = deepcopy(base_config)
        iters = []
        for kk in [10, 100, 1000, 10000, 100000, 1000000, 10000000]:
            for jj in [1, 2, 5]:
                iters.append(kk*jj)
        labelbase = current_config['config_common']['json_label']
        for ITER in iters:
            for dset in ['phototourism', 'pragueparks', 'googleurban']:
                try:
                   del current_config[f'config_{dset}_stereo']['geom']['error_type']
                   del current_config[f'config_{dset}_stereo']['geom']['degeneracy_check']
                except:
                   pass
                current_config[f'config_{dset}_stereo']['geom']['method'] = R
                current_config[f'config_{dset}_stereo']['geom']['threshold'] = final_ths[dset]
                current_config[f'config_{dset}_stereo']['geom']['max_iter'] = ITER
                current_config['config_common']['json_label']  = f'{labelbase}-{R}-iter-{ITER}'
            configs.append(deepcopy(current_config))
        with open(f'iter_ransac/iteration_{R}.json', 'w') as f:
            json.dump(configs, f, indent=2)
    for R in ['cmp-degensac-f']:
        res_dict = deepcopy(base_res_dict)
        for inl_th in inl_ths:
            res_fname = os.path.join('../image-matching-benchmark/packed-val', f'{hashname}-{R}-inlth-{inl_th}.json')
            try:
                with open(res_fname, 'r') as f:
                    results = json.load(f)
            except:
                 print (f'No results for {R} {inl_th}')
                 continue
            submission_name = results['config']['metadata']['method_name']
            res_dict[inl_th] = {}
            for dset in datasets:
                mAA = results[dset]['results']['allseq'][task]['run_avg'][metric]['mean']
                res_dict[dset][inl_th] = mAA
        final_ths = {}
        for i, dset in enumerate(datasets):
            inl_ths = []
            mAAs = []
            for inl_th, mAA in res_dict[dset].items():
                inl_ths.append(inl_th)
                mAAs.append(mAA)
            best_th_idx = np.argmax(np.array(mAAs))
            best_th = inl_ths[best_th_idx]
            best_mAA = mAAs[best_th_idx]
            print (mAAs)
            print (f'Best {R} {dset} mAA = {best_mAA:.4f} with inl_th = {best_th}')
            final_ths[dset] = best_th
        configs = []
        current_config = deepcopy(base_config)
        iters = []
        for kk in [10, 100, 1000, 10000, 100000, 1000000, 10000000]:
            for jj in [1, 2, 5]:
                iters.append(kk*jj)
        labelbase = current_config['config_common']['json_label']
        for ITER in iters:
            for dset in ['phototourism', 'pragueparks', 'googleurban']:
                current_config[f'config_{dset}_stereo']['geom']['threshold'] = final_ths[dset]
                current_config[f'config_{dset}_stereo']['geom']['method'] = R
                current_config[f'config_{dset}_stereo']['geom']['max_iter'] = ITER
                current_config['config_common']['json_label']  = f'{labelbase}-{R}-iter-{ITER}'
            configs.append(deepcopy(current_config))
        with open(f'iter_ransac/iteration_{R}.json', 'w') as f:
            json.dump(configs, f, indent=2)
    for R in ['cmp-degensac-f']:
        res_dict = deepcopy(base_res_dict)
        for inl_th in inl_ths:
            res_fname = os.path.join('../image-matching-benchmark/packed-val', f'{hashname}-nodegen-inlth-{inl_th}.json')
            try:
                with open(res_fname, 'r') as f:
                    results = json.load(f)
            except:
                 print (f'No results for {R} {inl_th}')
                 continue
            submission_name = results['config']['metadata']['method_name']
            res_dict[inl_th] = {}
            for dset in datasets:
                mAA = results[dset]['results']['allseq'][task]['run_avg'][metric]['mean']
                res_dict[dset][inl_th] = mAA
        final_ths = {}
        for i, dset in enumerate(datasets):
            inl_ths = []
            mAAs = []
            for inl_th, mAA in res_dict[dset].items():
                inl_ths.append(inl_th)
                mAAs.append(mAA)
            best_th_idx = np.argmax(np.array(mAAs))
            best_th = inl_ths[best_th_idx]
            best_mAA = mAAs[best_th_idx]
            print (f'Best nodegen {dset}  mAA = {best_mAA:.4f} with inl_th = {best_th}')
            final_ths[dset] = best_th
        configs = []
        current_config = deepcopy(base_config)
        labelbase = current_config['config_common']['json_label']
        iters = []
        for kk in [10, 100, 1000, 10000, 100000, 1000000, 10000000]:
            for jj in [1, 2, 5]:
                iters.append(kk*jj)
        for ITER in iters:
            for dset in ['phototourism', 'pragueparks', 'googleurban']:
                current_config[f'config_{dset}_stereo']['geom']['threshold'] = final_ths[dset]
                current_config[f'config_{dset}_stereo']['geom']['method'] = R
                current_config[f'config_{dset}_stereo']['geom']['degeneracy_check'] = False
                current_config[f'config_{dset}_stereo']['geom']['max_iter'] = ITER
                label = current_config['config_common']['json_label']
                current_config['config_common']['json_label']  = f'{labelbase}-nodegen-iter-{ITER}'
            configs.append(deepcopy(current_config))
        with open(f'iter_ransac/iteration_nodegen.json', 'w') as f:
            json.dump(configs, f, indent=2)
    
