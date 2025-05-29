import json
from obb_anns import OBBAnns
filepath = r"C:\Users\alexh\Desktop\ComputerVision\Hausarbeit\deepscores_v2_dense\ds2_dense\deepscores_train.json"
o = OBBAnns(filepath)
o.load_annotations()



with open(filepath, 'r') as f:
    data = json.load(f)

image_ids = [image['id'] for image in data['images']]
print("len(image_ids)", len(image_ids))
imgs, anns = o.get_img_ann_pair(image_ids[:5])