# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import csv
import pandas as pd

# th architecture to use
arch = 'resnet50'

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()


# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

"""
Folder structure:
content
 |
 +-- downloaded_videos
 |
 +-- frames
 |  |
 |  +-- folder with frames inside
 |
 +-- places365-nb_colab
 |  |
 |  +-- run_placesCNN_basic.py
"""

frame_folder = os.path.join(os.getcwd(), 'frames') #if only the file is uploaded
final_predictions = list()
video_df_colnames = pd.concat([pd.Series(['media_id', 'frame_nr']), pd.Series(classes)])

for video_folder in os.listdir('frames'):
    media_id = video_folder
    for frame in os.listdir(os.path.join(frame_folder, video_folder)):
        frame_nr = frame[frame.rfind('_')+1:-4]
        img_name = os.path.join(frame_folder, video_folder, frame)
        img = Image.open(img_name)
        input_img = V(centre_crop(img).unsqueeze(0))

        # forward pass
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        probs_as_list = probs.squeeze().tolist()
        final_predictions.append([media_id, frame_nr]+probs_as_list)

df = pd.DataFrame(final_predictions, columns=video_df_colnames)
df.to_csv('scene_prediction_values.csv', index=False)
