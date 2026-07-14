import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import RTDETR




if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/PWV.yaml')
    # model.load('') # loading pretrain weights


    model.train(data='dataset/mydata.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=4, # batchsize 不建议乱动，一般来说4的效果都是最好的
                workers=0, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                project='runs/train',
                name='pwv-detr',
                patience=20,  # 早停：20轮无提升则停止
                #seed=0,
                val=True,
                #save_period=5
                )