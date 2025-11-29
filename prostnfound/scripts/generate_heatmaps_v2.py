# prostnfound inference

from tqdm import tqdm
from src.transform import ProstNFoundTransform
from PIL import Image 
import numpy as np 
import json
import torch 
from medAI.modeling import create_model
from torch.utils.data import default_collate
from src.utils import render_heatmap
import os
from matplotlib import pyplot as plt


checkpoint_path = 'logs/pnf_optimum_finetune_v3/fold_0/best.pth'
case = "UA-003"
sd = torch.load(checkpoint_path, map_location='cpu')

transform = ProstNFoundTransform(image_size=512, mask_size=128)

from src.train import ProstNFoundMeta

model = create_model(sd['args']['model'], **sd['args']['model_kw'])
model = ProstNFoundMeta(model)
model.load_state_dict(sd['model'])
model.eval()


with open('/h/pwilson/projects/medAI/data/OPTIMUM/processed/processed/UA_sweeps/UA-011/UA-011-002/info.json') as f:
    info = json.load(f)


video_folder = '/h/pwilson/projects/medAI/data/OPTIMUM/processed/processed/UA_sweeps/UA-011/UA-011-002/bimg_video'
output_folder = 'UA-011-002_heatmaps'
os.makedirs(output_folder, exist_ok=True)

for path in tqdm(sorted(os.listdir(video_folder))): 

    image = np.array(Image.open(os.path.join(video_folder, path)))
    image = np.flipud(image)

    data = default_collate([transform(dict(bmode=image, psa=info['psa'], age=info['age']))])

    data = model(data, include_postprocessed_heatmaps=True)


    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    render_heatmap(
        ax, 
        np.flipud(image), 
        np.flipud(data['cancer_probs'][0][0].cpu()),
    )
    
    plt.savefig(os.path.join(output_folder, path), dpi=300, bbox_inches='tight')
    plt.close()
