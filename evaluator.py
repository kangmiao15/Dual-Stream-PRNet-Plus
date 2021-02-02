import numpy as np


def get_tp(gt_data, pred_data):
    return np.sum(gt_data[pred_data==1])


def get_tn(gt_data, pred_data):
    return np.sum(gt_data[pred_data==1]==0)


def get_fp(gt_data, pred_data, tp=None):
    if tp is None:
        tp = get_tp(gt_data, pred_data)
    return np.sum(pred_data) - tp


def get_fn(gt_data, pred_data, tp=None):
    if tp is None:
        tp = get_tp(gt_data, pred_data)
    return np.sum(gt_data) - tp


class EvalPrecision(object):
    def __init__(self):
        self.sum_score = 0
        self.count = 0

    def AddResult(self, predict, target):
        tp = get_tp(target, predict)
        fp = get_fp(target, predict, tp)
        if tp+fp == 0:
            return
        precision = 1.0*tp/(tp+fp)
        self.sum_score += precision
        self.count += 1
        return precision

    def Eval(self):
        if self.count > 0:
            return self.sum_score/self.count
        else:
            return 0


class EvalRecall(object):
    def __init__(self):
        self.num_hit = 0
        self.num_target = 0
        self.tp = 0
        self.fn = 0

    def AddResult(self, predict, target):
        tp = get_tp(target, predict)
        fn = get_fn(target, predict, tp)
        self.tp += tp
        self.fn += fn

    def Eval(self):
        recall = 1.0 * self.tp / (self.tp+self.fn+1)
        return recall


class EvalFscore(object):
    def __init__(self):
        pass

    def AddResult(self, predict, target):
        pass

    def Eval(self):
        pass


class EvalDiceScore(object):
    def __init__(self):
        self.sum_score = 0
        self.count = 0

    def AddResult(self, predict, target):
        tp = get_tp(target, predict)
        fp = get_fp(target, predict, tp)
        fn = get_fn(target, predict, tp)
        if tp*2.0+fn+fp == 0:
            return
        dice_score = tp * 2.0 / (tp * 2.0 + fn + fp)
        self.sum_score += dice_score
        self.count += 1
        return dice_score

    def Eval(self):
        if self.count > 0:
            return self.sum_score/self.count
        else:
            return 0
            
class EvalMeanDice(object):
    def __init__(self):
        self.score_list = [] 

    def AddResult(self, predict, target):
        tp = get_tp(target, predict)
        fp = get_fp(target, predict, tp)
        fn = get_fn(target, predict, tp)
        if tp*2.0+fn+fp == 0:
            return 
        dice_score = tp * 2.0 / (tp * 2.0 + fn + fp)
        self.score_list.append(dice_score)
        return self.score_list

    def Eval(self):
        return self.score_list

class EvalSensitivity(object):
    def __init__(self):
        self.sum_score = 0
        self.count = 0

    def AddResult(self, predict, target):
        tp = get_tp(target, predict)
        fn = get_fn(target, predict, tp)
        if tp+fn == 0:
            return
        recall = 1.0*tp/(tp+fn)
        self.sum_score += recall
        self.count += 1
        return recall

    def Eval(self):
        if self.count > 0:
            return self.sum_score/self.count
        else:
            return 0


class EvalDetectPrec(object):

    def __init__(self, iou=0.5):
        self.iou = iou
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def AddResult(self, predict, target):
        num_depth = predict.shape[0]
        predict = predict.reshape(num_depth, -1)
        target = target.reshape(num_depth, -1)
        area_predict = np.sum(predict)
        area_target = np.sum(target)
        area_inter = np.sum(np.logical_and(predict==1,target==1))
        iou = 1.0*area_inter/(area_predict+area_target-area_inter+1e-4)
        self.tp += 1 if iou >= self.iou else 0
        self.fp += 1 if iou < self.iou and area_predict > 0 else 0
        self.fn += 1 if iou < self.iou and area_target > 0 else 0
        return iou

    def Eval(self):
        precision = 1.0*self.tp/(self.tp+self.fp)
        return precision


class EvalDetectRecall(object):


    def __init__(self, iou=0.5):
        self.iou = iou
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def AddResult(self, predict, target):
        num_depth = predict.shape[0]
        predict = predict.reshape(num_depth, -1)
        target = target.reshape(num_depth, -1)
        area_predict = np.sum(predict)
        area_target = np.sum(target)
        area_inter = np.sum(np.logical_and(predict==1,target==1))
        iou = 1.0*area_inter/(area_predict+area_target-area_inter+1e-4)
        self.tp += 1 if iou >= self.iou else 0
        self.fp += 1 if iou < self.iou and area_predict > 0 else 0
        self.fn += 1 if iou < self.iou and area_target > 0 else 0
        return iou

    def Eval(self):
        recall = 1.0*self.tp/(self.tp+self.fn)
        return recall

class EvalTarRegErr(object):
    def __init__(self):
        self.error_list = [] 

    def AddResult(self, points_1, points_2):
        """ 
        computing Target Registration Error for each landmark pair
        :param ndarray points_1: set of points
        :param ndarray points_2: set of points
        :return ndarray: list of errors of size min nb of points
        np.random.seed(0)
        compute_tre(np.random.random((6, 2)),
                   np.random.random((9, 2)))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        array([ 0.21...,  0.70...,  0.44...,  0.34...,  0.41...,  0.41...])
        """
        nb_common = min([len(pts) for pts in [points_1, points_2]
                        if pts is not None])
        assert nb_common > 0, 'no common landmarks for metric'
        points_1 = np.asarray(points_1)[:nb_common]
        points_2 = np.asarray(points_2)[:nb_common]
        # convert to world corordination(mm) with space resolution
        points_1 = points_1 * np.array([2.5,1.0,1.0])
        points_2 = points_2 * np.array([2.5,1.0,1.0])
        # diffs = np.mean(np.sqrt(np.sum(np.power(points_1 - points_2, 2), axis=1)))
        errors = [np.linalg.norm(points_1 - points_2,axis = 1)]
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        self.error_list.append((mean_error, std_error))
        return self.error_list

    def Eval(self):
        return self.error_list

if __name__ == '__main__':
    p1 = np.array(([0,0,1], [1,0,1]))
    p2 = np.array(([1,1,1], [0,1,1]))
    tre = EvalTarRegErr()
    tre.AddResult(p1, p2)
