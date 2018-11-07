import json
import random
from PIL import Image
import os
import numpy as np
def get_image(data='train',shuffle=True):
    data_path='data/brainwash/train_boxes.json'
    if(data=='test'):
        data_path='data/brainwash/test_boxes.json'
    elif(data=='val'):
        data_path='data/brainwash/val_boxes.json'
    with open(data_path) as train_file:
        data_list=json.load(train_file)
        length=len(data_list)
        i=0
        while True:
            if(shuffle):
                i=random.randint(0,length-1)
            else:
                i=(i+1)%length
            image = Image.open(os.path.join('data/brainwash/', data_list[i]['image_path']))
            # plt.imshow(image)
            image_arr = np.array(image)
            image_arr = image_arr.reshape([1, 640, 480, 3])
            rects=data_list[i]['rects']
            if (len(rects) > 0):
                y = [rects[0]['x1'], rects[0]['x2'], rects[0]['y1'], rects[0]['y2']]
            else:
                y = [0, 0, 0, 0]
            yield image_arr,y

# f=get_image()
# d=f.__next__()