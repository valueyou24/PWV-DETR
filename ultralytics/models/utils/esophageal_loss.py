# ultralytics/models/utils/esophageal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import RTDETRDetectionLoss
from ultralytics.utils.metrics import bbox_iou


class EsophagealAwareLoss(RTDETRDetectionLoss):
    """
    食管癌感知损失函数 - 完全兼容RTDETRDetectionLoss
    创新点：密度感知 + 解剖位置先验
    """

    def __init__(self, nc=1,
                 use_density_aware=True,
                 use_anatomical_prior=True,
                 density_weight=0.3,
                 anatomy_weight=0.2,
                 **kwargs):
        """
        初始化食管癌感知损失函数

        Args:
            nc (int): 类别数
            use_density_aware (bool): 是否使用密度感知
            use_anatomical_prior (bool): 是否使用解剖位置先验
            density_weight (float): 密度损失权重
            anatomy_weight (float): 解剖损失权重
        """
        super().__init__(nc=nc, **kwargs)

        self.use_density_aware = use_density_aware
        self.use_anatomical_prior = use_anatomical_prior
        self.density_weight = density_weight
        self.anatomy_weight = anatomy_weight

        # 食管解剖位置先验 (基于医学知识)
        self.esophagus_regions = {
            'cervical': [0.1, 0.3],  # 颈段 - 较难检测
            'thoracic_upper': [0.3, 0.5],  # 胸上段
            'thoracic_middle': [0.5, 0.7],  # 胸中段 - 最常见
            'thoracic_lower': [0.7, 0.9]  # 胸下段
        }
        self.region_weights = {
            'cervical': 1.3,  # 颈段权重较高
            'thoracic_upper': 1.1,  # 胸上段
            'thoracic_middle': 0.8,  # 胸中段权重较低（易检测）
            'thoracic_lower': 1.0  # 胸下段
        }

        # CT密度范围 (HU值)
        self.hu_ranges = {
            'low': [-100, 20],  # 低密度 (气体、脂肪)
            'soft_tissue': [20, 50],  # 软组织
            'barium': [100, 500],  # 钡餐造影
            'high': [500, 1000]  # 高密度 (钙化)
        }

        print("🎯 食管癌感知损失函数初始化完成")
        print(f"  密度感知: {use_density_aware}, 权重: {density_weight}")
        print(f"  解剖先验: {use_anatomical_prior}, 权重: {anatomy_weight}")

    def _compute_density_weights(self, batch, gt_bboxes):
        """计算基于CT密度的权重"""
        if not self.use_density_aware:
            return None

        try:
            images = batch['img']
            batch_densities = []

            for i, bboxes in enumerate(gt_bboxes):
                if bboxes is None or len(bboxes) == 0:
                    batch_densities.append(None)
                    continue

                densities = []
                for bbox in bboxes:
                    try:
                        # bbox是归一化坐标 [x1, y1, x2, y2]
                        x1, y1, x2, y2 = bbox

                        # 转换为像素坐标
                        h, w = images.shape[2], images.shape[3]
                        x1_pix = int(x1 * w);
                        y1_pix = int(y1 * h)
                        x2_pix = int(x2 * w);
                        y2_pix = int(y2 * h)

                        # 确保坐标有效
                        x1_pix = max(0, x1_pix);
                        y1_pix = max(0, y1_pix)
                        x2_pix = min(w - 1, x2_pix);
                        y2_pix = min(h - 1, y2_pix)

                        if x2_pix <= x1_pix or y2_pix <= y1_pix:
                            densities.append(0.5)
                            continue

                        # 提取病灶区域并计算密度特征
                        lesion_region = images[i, :, y1_pix:y2_pix, x1_pix:x2_pix]
                        if lesion_region.numel() == 0:
                            densities.append(0.5)
                            continue

                        hu_mean = torch.mean(lesion_region).item()

                        # 基于HU值分配密度权重
                        if hu_mean < self.hu_ranges['low'][1]:
                            density_score = 0.3  # 低密度，难检测
                        elif hu_mean < self.hu_ranges['soft_tissue'][1]:
                            density_score = 0.6  # 软组织
                        elif hu_mean < self.hu_ranges['barium'][1]:
                            density_score = 0.9  # 钡餐，易检测
                        else:
                            density_score = 0.7  # 高密度

                        densities.append(density_score)

                    except Exception:
                        densities.append(0.5)  # 出错时使用默认值

                if densities:
                    batch_densities.append(torch.tensor(densities, device=images.device))
                else:
                    batch_densities.append(None)

            return batch_densities

        except Exception as e:
            print(f"密度权重计算失败: {e}")
            return None

    def _compute_anatomy_weights(self, gt_bboxes, device):
        """计算基于解剖位置的权重"""
        if not self.use_anatomical_prior:
            return None

        try:
            if gt_bboxes is None or len(gt_bboxes) == 0:
                return None

            prior_weights = []

            for bboxes in gt_bboxes:
                if bboxes is None or len(bboxes) == 0:
                    prior_weights.append(None)
                    continue

                weights = torch.ones(len(bboxes), device=device)

                # 计算每个bbox的中心y坐标 (归一化)
                centers_y = (bboxes[:, 1] + bboxes[:, 3]) / 2

                # 基于食管解剖位置分配权重
                for region_name, (y_min, y_max) in self.esophagus_regions.items():
                    region_mask = (centers_y >= y_min) & (centers_y < y_max)
                    weights[region_mask] = self.region_weights[region_name]

                prior_weights.append(weights)

            return prior_weights

        except Exception as e:
            print(f"解剖权重计算失败: {e}")
            return None

    def _get_loss_giou_esophageal(self, pred_bboxes, gt_bboxes, batch):
        """食管癌感知的GIoU损失"""
        if len(gt_bboxes) == 0:
            return {
                'loss_giou_esophageal': torch.tensor(0., device=self.device),
                'loss_density': torch.tensor(0., device=self.device),
                'loss_anatomy': torch.tensor(0., device=self.device)
            }

        # 基础GIoU损失
        base_giou = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
        base_giou_loss = base_giou.sum() / len(gt_bboxes)

        # 食管癌专用损失组件
        esophageal_losses = {
            'loss_giou_esophageal': base_giou_loss * self.loss_gain['giou'],
            'loss_density': torch.tensor(0., device=self.device),
            'loss_anatomy': torch.tensor(0., device=self.device)
        }

        # 密度感知损失
        if self.use_density_aware:
            density_weights = self._compute_density_weights(batch, [gt_bboxes])
            if density_weights and density_weights[0] is not None:
                weights = 1.2 - density_weights[0]  # 密度越低，权重越高
                weights = torch.clamp(weights, 0.5, 1.5)
                density_loss = (base_giou * weights).mean()
                esophageal_losses['loss_density'] = density_loss * self.density_weight

        # 解剖位置先验损失
        if self.use_anatomical_prior:
            anatomy_weights = self._compute_anatomy_weights([gt_bboxes], gt_bboxes.device)
            if anatomy_weights and anatomy_weights[0] is not None:
                anatomy_loss = (base_giou * anatomy_weights[0]).mean()
                esophageal_losses['loss_anatomy'] = anatomy_loss * self.anatomy_weight

        return esophageal_losses

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                  masks=None, gt_mask=None, postfix='', match_indices=None, batch=None):
        """重写_get_loss方法以集成食管癌感知"""
        # 调用父类的匹配和基础损失计算
        if match_indices is None:
            match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups)

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes_matched, gt_bboxes_matched = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes_matched):
            gt_scores[idx] = bbox_iou(pred_bboxes_matched.detach(), gt_bboxes_matched, xywh=True).squeeze(-1)

        # 基础损失
        loss = {}
        loss.update(self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes_matched), postfix))
        loss.update(self._get_loss_bbox(pred_bboxes_matched, gt_bboxes_matched, postfix))

        # 食管癌感知的GIoU损失
        if len(gt_bboxes_matched) > 0 and batch is not None:
            esophageal_giou_losses = self._get_loss_giou_esophageal(pred_bboxes_matched, gt_bboxes_matched, batch)
            loss.update(esophageal_giou_losses)

            # 更新总GIoU损失
            if 'loss_giou' + postfix in loss:
                loss['loss_giou' + postfix] = loss['loss_giou' + postfix] + \
                                              esophageal_giou_losses['loss_density'] + \
                                              esophageal_giou_losses['loss_anatomy']

        return loss

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        """
        前向传播 - 集成食管癌感知损失
        """
        pred_bboxes, pred_scores = preds

        # 基础损失计算
        total_loss = super().forward(pred_bboxes, pred_scores, batch)

        # 食管癌专用损失（只在训练时计算）
        if self.training:
            esophageal_specific_losses = {}

            # 对每一层解码器输出计算食管癌感知损失
            for i, (pred_bbox, pred_score) in enumerate(zip(pred_bboxes, pred_scores)):
                if i == len(pred_bboxes) - 1:  # 只对最后一层计算，避免重复
                    esophageal_loss = self._get_loss(
                        pred_bbox, pred_score, batch['bboxes'], batch['cls'],
                        batch['gt_groups'], postfix='', match_indices=None, batch=batch
                    )

                    # 提取食管癌专用损失
                    for key in ['loss_density', 'loss_anatomy']:
                        if key in esophageal_loss:
                            esophageal_specific_losses[key] = esophageal_loss[key]

            # 合并损失
            total_loss.update(esophageal_specific_losses)

            # 在TensorBoard中显示食管癌专用损失
            if self.use_density_aware and 'loss_density' in esophageal_specific_losses:
                total_loss['loss_density'] = esophageal_specific_losses['loss_density']
            if self.use_anatomical_prior and 'loss_anatomy' in esophageal_specific_losses:
                total_loss['loss_anatomy'] = esophageal_specific_losses['loss_anatomy']

        return total_loss