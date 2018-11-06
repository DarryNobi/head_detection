import json
import random
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
                i=random.randint(0,length)
            else:
                i=(i+1)%length
            yield data_list[i]['image_path'],data_list[i]['rects']

# f=get_image()
# d=f.__next__()