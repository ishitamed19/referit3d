import os
import numpy as np

ROOT_DIR = '/projects/katefgroup/language_grounding/scans/group_free_pred_bboxes' # 81 
ROOT_DIR = '/projects/katefgroup/language_grounding/scans/group_free_pred_bboxes_03' # 101

with open('./referit3d/data/scannet/splits/official/v2/scannetv2_val.txt', "rb") as f:
    bytelist = f.read().splitlines() 
    all_scan_ids = [x.decode('utf-8') for x in bytelist]

max_ob = 0
for scan_id in all_scan_ids:
    if scan_id.endswith('00'):
        bbox = np.load(os.path.join(ROOT_DIR, scan_id + '_pred_boxes.npy'))
        max_ob = max(max_ob, bbox.shape[0])

print(max_ob)