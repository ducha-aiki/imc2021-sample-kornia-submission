import json
from copy import deepcopy

metadata_dict =  {
    "publish_anonymously": False,
    "authors": "Dmytro Mishkin",
    "contact_email": "ducha.aiki@gmail.com",
    "method_name": "Testing OpenCV RANSAC",
    "method_description": 
    r"""OpeCV RootSIFT matched using the built-in matcher (bidirectional filter with the 'both' strategy, 
    hopefully optimal inlier and ratio test thresholds) with Various OpenCV RANSAC""",
    "link_to_website": "https://docs.opencv.org/master/d1/df1/md__build_master-contrib_docs-lin64_opencv_doc_tutorials_calib3d_usac.html",
    "link_to_pdf": ""
}

config_common_dict =  {"json_label": "tuning-",
    "keypoint": "sift-lowth",
    "descriptor": "rootsift",
    "num_keypoints": 8000}


matcher_template_dict = {
     "method": "nn",
     "distance": "L2",
     "flann": True,
     "num_nn": 1,
     "filtering": {
         "type": "snn_ratio_pairwise",
         "threshold": 0.85,
     },
     "symmetric": {
         "enabled": True,
         "reduce": "both",
     }
}

geom_template_dict =  {"method": "cmp-degensac-f",
                "threshold": 0.5,
                "confidence": 0.999999,
                "max_iter": 100000,
                "error_type": "sampson",
                "degeneracy_check": True,
            }

base_config =  {
    "metadata": metadata_dict,
    "config_common": config_common_dict,
    "config_phototourism_stereo": {
        "use_custom_matches": False,
        "matcher": deepcopy(matcher_template_dict),
        "outlier_filter": { "method": "none" },
        "geom": deepcopy(geom_template_dict)
        },
    "config_phototourism_multiview": {
        "use_custom_matches": False,
        "matcher": deepcopy(matcher_template_dict),
        "outlier_filter": { "method": "none" },
        "colmap": {}},
    
    "config_pragueparks_stereo": {
        "use_custom_matches": False,
        "matcher": deepcopy(matcher_template_dict),
        "outlier_filter": { "method": "none" },
        "geom": deepcopy(geom_template_dict)
        },
    "config_pragueparks_multiview": {
        "use_custom_matches": False,
        "matcher": deepcopy(matcher_template_dict),
        "outlier_filter": { "method": "none" },
        "colmap": {}},
    "config_googleurban_stereo": {
        "use_custom_matches": False,
        "matcher": deepcopy(matcher_template_dict),
        "outlier_filter": { "method": "none" },
        "geom": deepcopy(geom_template_dict)
        },
    "config_googleurban_multiview": {
        "use_custom_matches": False,
        "matcher": deepcopy(matcher_template_dict),
        "outlier_filter": { "method": "none" },
        "colmap": {}}
}

if __name__ == '__main__':
    inl_ths = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    ransacs = ['cv2-usacdef-f',
               'cv2-usacmagsac-f',
               'cv2-usacfast-f',
               'cv2-usacaccurate-f', 'cv2-ransac-f']
    for R in ransacs:
        configs = []
        for inl_th in inl_ths:
            current_config = deepcopy(base_config)
            for dset in ['phototourism', 'pragueparks', 'googleurban']:
                current_config[f'config_{dset}_stereo']['geom']['threshold'] = inl_th
                current_config[f'config_{dset}_stereo']['geom']['method'] = R
                del current_config[f'config_{dset}_stereo']['geom']['error_type']
                del current_config[f'config_{dset}_stereo']['geom']['degeneracy_check']
            label = current_config['config_common']['json_label'] 
            current_config['config_common']['json_label']  = f'{label}-{R}-inlth-{inl_th}'
            configs.append(current_config)
        with open(f'{R}_tuning.json', 'w') as f:
            json.dump(configs, f, indent=2)
    configs = []
    R = 'cmp-degensac-f'
    for inl_th in inl_ths:
        current_config = deepcopy(base_config)
        for dset in ['phototourism', 'pragueparks', 'googleurban']:
            current_config[f'config_{dset}_stereo']['geom']['threshold'] = inl_th
        label = current_config['config_common']['json_label'] 
        current_config['config_common']['json_label']  = f'{label}-{R}-inlth-{inl_th}'
        configs.append(deepcopy(current_config))
        for dset in ['phototourism', 'pragueparks', 'googleurban']:
            current_config[f'config_{dset}_stereo']['geom']['degeneracy_check'] = False
        current_config['config_common']['json_label']  = f'{label}-nodegen-inlth-{inl_th}'
        configs.append(deepcopy(current_config))
    with open(f'{R}_tuning.json', 'w') as f:
        json.dump(configs, f, indent=2)
    
