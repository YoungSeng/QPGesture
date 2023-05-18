import numpy as np
from constant import NUM_AUDIO_FEAT, NUM_BODY_FEAT_FRAMES, BODY_FEAT_IDX


def create_control_filter(feat_train, control_type=None, n_aud_feat=NUM_AUDIO_FEAT):
    num_seq, _, num_frm = feat_train.shape
    control_mask = np.ones((num_seq, num_frm))
    num_joint_feat = len(BODY_FEAT_IDX)

    if control_type is None:
        return control_mask

    body_feat = feat_train.transpose((0, 2, 1))
    body_feat = body_feat[:, :, n_aud_feat:]
    body_feat = body_feat.reshape((num_seq, num_frm, NUM_BODY_FEAT_FRAMES, -1))
    body_feat = body_feat.reshape((num_seq, num_frm, NUM_BODY_FEAT_FRAMES, num_joint_feat, 3))
    
    # the index of the left wrist joint in BODY_FEAT_IDX is 3
    lwrist_height = body_feat[:, :, 0, 3, 1]

    height_list = []
    for i in range(feat_train.shape[0]):
        for j in range(0, feat_train.shape[2]):
            height_list.append(lwrist_height[i, j] * -1)

    quantile_high = np.quantile(height_list, 0.85)
    quantile_low = np.quantile(height_list, 0.15)

    if control_type == "hand_high":
        for i in range(num_seq):
            for j in range(num_frm):
                # mask out frames where hand height is below threshold
                # the y-axis is multiplied by -1 to flip it upward
                if lwrist_height[i, j] * -1 < quantile_high:              
                    control_mask[i, j] = 0
    elif control_type == "hand_low":
        for i in range(num_seq):
            for j in range(num_frm):
                # mask out frames where hand height is above threshold
                # the y-axis is multiplied by -1 to flip it upward
                if lwrist_height[i, j] * -1 > quantile_low:              
                    control_mask[i, j] = 0 
    else:
        control_mask = np.ones((num_seq, num_frm))                              
    
    return control_mask