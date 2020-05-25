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
csv_header = classes
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

# load the test image
#frame_folder = os.path.join('..', os.path.dirname(os.getcwd()), 'frames') if it is in a folder
frame_folder = os.path.join(os.getcwd(), 'frames') #if only the file is uploaded
final_predictions = list()

for video_folder in os.listdir('frames'):
    for frame in os.listdir(os.path.join(frame_folder, video_folder)):

        img_name = os.path.join(frame_folder, video_folder, frame)
        img = Image.open(img_name)
        input_img = V(centre_crop(img).unsqueeze(0))

        # forward pass
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        probs_as_list = probs.squeeze().tolist()
        final_predictions.append(probs_as_list)

        #final_predictions.append(list(zip(classes, probs_as_list))) for testing
#print(final_predictions) also for tsting

#write predictions to file
csv_file = 'scene_prediction_values.csv'

with open(csv_file, 'w+', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_header)
    writer.writerows(final_predictions)
