# NMS(Non-maximum suppression)
# http://pynote.hatenablog.com/entry/opencv-non-maximum-suppression
物体検出を行うと1つの物体に対して複数回検出されるということがある。
これらを1つの短形に統合する Non Maximum Suppression という処理がよく用いられる。

NMSとはスコアの高いBounding-boxのみを残す手法であり、アルゴリズムは以下の通り。

1. Boundinb-boxの集合Bをスコアが高い順にソートする。
2. スコアが最大のものをb0とする。
3. b0と他のBounding-boxのIoUを計算する。IoUが閾値t以上のBounding-boxをBから削除する。B0は出力する集合Rに加え、Bから削除する。
4. 2から3をBがなくなるまで行う。
5. Rを出力する。

def nms(_bboxes, iou_th=0.5, select_num=None, prob_th=None):
    #
    # Non Maximum Suppression
    #
    # Argument
    #  bboxes(Nx5) ... [bbox-num, 5(leftTopX,leftTopY,w,h, score)]
    #  iou_th([float]) ... threshold for iou between bboxes.
    #  select_num([int]) ... max number for choice bboxes. If None, this is unvalid.
    #  prob_th([float]) ... probability threshold to choice. If None, this is unvalid.
    # Return
    #  inds ... choced indices for bboxes
    #

    bboxes = _bboxes.copy()
    
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    
    # Sort by bbox's score. High -> Low
    sort_inds = np.argsort(bboxes[:, -1])[::-1]

    processed_bbox_ind = []
    return_inds = []

    unselected_inds = sort_inds.copy()
    
    while len(unselected_inds) > 0:
        process_bboxes = bboxes[unselected_inds]
        argmax_score_ind = np.argmax(process_bboxes[::, -1])
        max_score_ind = unselected_inds[argmax_score_ind]
        return_inds += [max_score_ind]
        unselected_inds = np.delete(unselected_inds, argmax_score_ind)

        base_bbox = bboxes[max_score_ind]
        compare_bboxes = bboxes[unselected_inds]
        
        base_x1 = base_bbox[0]
        base_y1 = base_bbox[1]
        base_x2 = base_bbox[2] + base_x1
        base_y2 = base_bbox[3] + base_y1
        base_w = np.maximum(base_bbox[2], 0)
        base_h = np.maximum(base_bbox[3], 0)
        base_area = base_w * base_h

        # compute iou-area between base bbox and other bboxes
        iou_x1 = np.maximum(base_x1, compare_bboxes[:, 0])
        iou_y1 = np.maximum(base_y1, compare_bboxes[:, 1])
        iou_x2 = np.minimum(base_x2, compare_bboxes[:, 2] + compare_bboxes[:, 0])
        iou_y2 = np.minimum(base_y2, compare_bboxes[:, 3] + compare_bboxes[:, 1])
        iou_w = np.maximum(iou_x2 - iou_x1, 0)
        iou_h = np.maximum(iou_y2 - iou_y1, 0)
        iou_area = iou_w * iou_h

        compare_w = np.maximum(compare_bboxes[:, 2], 0)
        compare_h = np.maximum(compare_bboxes[:, 3], 0)
        compare_area = compare_w * compare_h

        # bbox's index which iou ratio over threshold is excluded
        all_area = compare_area + base_area - iou_area
        iou_ratio = np.zeros((len(unselected_inds)))
        iou_ratio[all_area < 0.9] = 0.
        _ind = all_area >= 0.9
        iou_ratio[_ind] = iou_area[_ind] / all_area[_ind]
        
        unselected_inds = np.delete(unselected_inds, np.where(iou_ratio >= iou_th)[0])

    if prob_th is not None:
        preds = bboxes[return_inds][:, -1]
        return_inds = np.array(return_inds)[np.where(preds >= prob_th)[0]].tolist()
        
    # pick bbox's index by defined number with higher score
    if select_num is not None:
        return_inds = return_inds[:select_num]

    return return_inds