import json
from copy import deepcopy

metadata_dict =  {
    "publish_anonymously": False,
    "authors": "Dmytro Mishkin, Milan Pultar and kornia team",
    "contact_email": "ducha.aiki@gmail.com",
    "method_name": "CV-DoG-AffNet-HardNet8 (kornia)",
    "method_description": 
    r"""OpeCV SIFT keypoints 8000 features, followed by the AffNet normalization 
    and HardNet8 descriptor as implemented in kornia.
    Matched using the built-in matcher (bidirectional filter with the 'both' strategy, 
    hopefully optimal inlier and ratio test thresholds) with DEGENSAC""",
    "link_to_website": "https://github.com/kornia/kornia",
    "link_to_pdf": "https://arxiv.org/abs/2007.09699"
}

config_common_dict =  {"json_label": "dog-affnet-hardnet8-degensac",
    "keypoint": "cv2dog",
    "descriptor": "affnethardnet8",
    "num_keypoints": 8000}


matcher_template_dict = {
     "method": "nn",
     "distance": "L2",
     "flann": True,
     "num_nn": 1,
     "filtering": {
         "type": "snn_ratio_pairwise",
         "threshold": 0.90,
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
    with open('base_config.json', 'w') as f:
        json.dump([base_config], f, indent=2)
        inl_ths = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    configs = []
    for inl_th in inl_ths:
        current_config = deepcopy(base_config)
        for dset in ['phototourism', 'pragueparks', 'googleurban']:
            current_config[f'config_{dset}_stereo']['geom']['threshold'] = inl_th
        label = current_config['config_common']['json_label'] 
        current_config['config_common']['json_label']  = f'{label}-inlth-{inl_th}'
        configs.append(current_config)
    with open('ransac_tuning.json', 'w') as f:
        json.dump(configs, f, indent=2)
