# test_loss_directly.py
import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, 'D:/DeepMed/rtdert_relate/RTDETR-main')


def test_loss_directly():
    print("=== 直接测试损失函数 ===")

    try:
        from ultralytics.models.utils.esophageal_loss import EsophagealCTDualLoss

        # 直接创建损失函数实例
        print("创建损失函数实例...")
        loss_fn = EsophagealCTDualLoss(nc=1)
        print(f"✅ 损失函数创建成功: {type(loss_fn)}")

        # 检查属性
        print("\n检查属性:")
        for attr in ['use_density_aware', 'use_anatomical_prior', 'density_loss', 'anatomy_loss']:
            if hasattr(loss_fn, attr):
                value = getattr(loss_fn, attr)
                print(f"  ✅ {attr}: {value}")
            else:
                print(f"  ❌ {attr}: 不存在")

        # 创建模拟数据
        print("\n创建模拟数据...")
        # 模拟 preds: (pred_bboxes, pred_scores)
        pred_bboxes = [torch.randn(1, 10, 4)]  # [层数, 批次, 查询数, 4]
        pred_scores = [torch.randn(1, 10, 1)]  # [层数, 批次, 查询数, 类别数]
        preds = (pred_bboxes, pred_scores)

        # 模拟 batch
        batch = {
            'img': torch.randn(1, 3, 640, 640),  # 1张图像
            'cls': [torch.tensor([0])],  # 类别
            'bboxes': [torch.tensor([[0.1, 0.1, 0.3, 0.3]])],  # 边界框
            'batch_idx': torch.tensor([0]),  # 批次索引
        }

        print("模拟数据创建成功:")
        print(f"  pred_bboxes: {pred_bboxes[0].shape}")
        print(f"  pred_scores: {pred_scores[0].shape}")
        print(f"  batch['img']: {batch['img'].shape}")
        print(f"  batch['bboxes']: 长度={len(batch['bboxes'])}, 形状={batch['bboxes'][0].shape}")

        # 测试损失计算
        print("\n测试损失计算...")
        loss_fn.train()  # 设置为训练模式
        loss_dict = loss_fn(preds, batch)

        print("✅ 损失计算成功")
        print("损失字典:")
        for k, v in loss_dict.items():
            if hasattr(v, 'item'):
                print(f"  {k}: {v.item():.4f}")
            else:
                print(f"  {k}: {v}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_individual_losses():
    print("\n=== 测试单个损失组件 ===")

    try:
        from ultralytics.models.utils.esophageal_loss import DensityAwareLoss, AnatomicalPriorLoss

        # 测试密度感知损失
        print("测试密度感知损失...")
        density_loss = DensityAwareLoss()

        # 模拟数据
        pred_bboxes = torch.randn(1, 2, 4)  # [批次, 查询数, 4]
        gt_bboxes = [torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]])]
        batch = {
            'img': torch.randn(1, 3, 640, 640),
            'bboxes': gt_bboxes
        }

        loss = density_loss(pred_bboxes, gt_bboxes, batch, 1.0)
        print(f"✅ 密度感知损失: {loss.item():.4f}")

        # 测试解剖位置先验损失
        print("测试解剖位置先验损失...")
        anatomy_loss = AnatomicalPriorLoss()

        loss = anatomy_loss(pred_bboxes, gt_bboxes, batch)
        print(f"✅ 解剖位置先验损失: {loss.item():.4f}")

    except Exception as e:
        print(f"❌ 单个损失测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_loss_directly()
    test_individual_losses()