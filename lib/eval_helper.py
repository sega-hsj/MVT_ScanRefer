import os
import sys
import torch
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from utils.box_util import get_3d_box, box3d_iou, get_3d_box_batch, box3d_iou_batch

from utils.util import construct_bbox_corners


def get_eval(data_dict, config):
    """ Loss functions
    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
        post_processing: config dict
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """

    lang_scores = data_dict["lang_scores"]
    lang_cls_pred = torch.argmax(lang_scores, dim=1)
    batch_size = lang_scores.shape[0]

    data_dict["lang_acc"] = (lang_cls_pred == data_dict["object_cat"]).float().mean()


    scores = data_dict['scores']
    # attribute_scores = data_dict['attribute_scores']
    # relation_scores = data_dict['relation_scores']
    # scene_scores = data_dict['scene_scores']

    pred_obb_batch = data_dict['pred_obb_batch']
    cluster_labels = data_dict['cluster_label']

    ref_center_label = data_dict["ref_center_label"].detach().cpu().numpy()
    ref_heading_class_label = data_dict["ref_heading_class_label"].detach().cpu().numpy()
    ref_heading_residual_label = data_dict["ref_heading_residual_label"].detach().cpu().numpy()
    ref_size_class_label = data_dict["ref_size_class_label"].detach().cpu().numpy()
    ref_size_residual_label = data_dict["ref_size_residual_label"].detach().cpu().numpy()

    ref_gt_obb = config.param2obb_batch(ref_center_label, ref_heading_class_label, ref_heading_residual_label,
                                        ref_size_class_label, ref_size_residual_label)

    ious = []
    pred_bboxes = []
    gt_bboxes = []
    ref_acc = []
    multiple = []
    others = []
    start_idx = 0
    num_missed = 0


    cluster_preds = torch.argmax(scores, dim=-1)
    targets = torch.argmax(cluster_labels, dim=-1)
    ref_acc = (cluster_preds == targets).float()
    B = cluster_preds.shape[0]
    pred_obbs = []
    for idx in range(B):
        pred_obbs = pred_obb_batch[idx]
        gt_obb = ref_gt_obb[idx]
        cluster_pred = cluster_preds[idx]
        pred_obb = pred_obbs[cluster_pred].cpu().numpy()
        
        pred_bbox = get_3d_box(pred_obb[3:6], pred_obb[6], pred_obb[0:3])
        gt_bbox = get_3d_box(gt_obb[3:6], gt_obb[6], gt_obb[0:3])
        iou = box3d_iou(pred_bbox, gt_bbox)
        ious.append(iou)

        pred_bbox = construct_bbox_corners(pred_obb[0:3], pred_obb[3:6])
        gt_bbox = construct_bbox_corners(gt_obb[0:3], gt_obb[3:6])
        pred_bboxes.append(pred_bbox)
        gt_bboxes.append(gt_bbox)

        multiple.append(data_dict["unique_multiple"][idx].item())
        flag = 1 if data_dict["object_cat"][idx] == 17 else 0
        others.append(flag)

    data_dict['ref_acc'] = ref_acc.cpu().numpy()

    data_dict["ref_iou"] = ious
    data_dict["ref_iou_rate_0.25"] = np.array(ious)[np.array(ious) >= 0.25].shape[0] / np.array(ious).shape[0]
    data_dict["ref_iou_rate_0.5"] = np.array(ious)[np.array(ious) >= 0.5].shape[0] / np.array(ious).shape[0]

    # data_dict["seg_acc"] = torch.ones(1)[0].cuda()
    data_dict["ref_multiple_mask"] = multiple
    data_dict["ref_others_mask"] = others
    data_dict["pred_bboxes"] = pred_bboxes
    data_dict["gt_bboxes"] = gt_bboxes

    return data_dict
