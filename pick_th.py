import os
import numpy as np
import json

inl_ths = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]

if __name__ == '__main__':
    hashname='dog-affnet-hardnet8-degensac'
    res_dict = {}
    datasets = ['phototourism', 'pragueparks', 'googleurban']
    for dset in datasets:
        res_dict[dset] = {}

    task = 'stereo'
    metric = 'qt_auc_10_th_0.1'
    for inl_th in inl_ths:
        res_fname = os.path.join('../image-matching-benchmark/packed-val', f'{hashname}-inlth-{inl_th}.json')
        try:
            with open(res_fname, 'r') as f:
                results = json.load(f)
        except:
            continue
        submission_name = results['config']['metadata']['method_name']
        res_dict[inl_th] = {}
        for dset in datasets:
            mAA = results[dset]['results']['allseq'][task]['run_avg'][metric]['mean']
            res_dict[dset][inl_th] = mAA
    #fig, ax = plt.subplots(figsize=(5,5))
    colors = ['r','b','k']
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
        print (f'Best {dset} mAA = {best_mAA:.4f} with inl_th = {best_th}')
        #ax.plot(inl_ths, mAAs, label=dset, color=colors[i])
        #ax.plot(best_th, best_mAA, label = f'{dset}-best', marker='x', linestyle='', color=colors[i])
        final_ths[dset] = best_th
    from create_base_config import *
    configs = []
    current_config = deepcopy(base_config)
    for dset in ['phototourism', 'pragueparks', 'googleurban']:
        current_config[f'config_{dset}_stereo']['geom']['threshold'] = final_ths[dset]
        # I did a little bit of tuning offline for multiview, so we will put it here
        current_config[f'config_{dset}_multiview']['matcher']['filtering']['threshold'] = 0.95
    current_config['metadata']['method_name'] = 'KORNIA TUTORIAL CV-DoG-AffNet-HardNet8'

    label = current_config['config_common']['json_label'] 
    current_config['config_common']['json_label']  = f'{label}'
    configs.append(current_config)

    print (current_config)
    with open('final_submission.json', 'w') as f:
        json.dump(configs, f, indent=2)
    #ax.legend()
    #ax.set_ylabel('mAA')
    #ax.set_xlabel('DEGENSAC inlier threshold')
    
