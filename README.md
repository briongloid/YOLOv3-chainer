# YOLOv3-chainer

- Python == 3.5.2

## cloneから仮想環境まで

```
git clone https://github.com/briongloid/YOLOv3-chainer.git
cd YOLOv3-chainer
python3 -m venv venv
. venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

cupyのインストール

```
pip install cupy
```

## Pascal VOCデータセットでDarknet53の学習

```
python darknet53_voc_train.py
```

GPUを使って学習

```
python darknet53_voc_train.py -g 0
```

複数のGPUを使って学習

```
python darknet53_voc_train.py -g 0 1
```

## Pascal VOCデータセットでYOLOv3の学習

```
python yolov3_voc_train.py
```

学習済みDarknet53と複数GPUを使った学習

```
python yolov3_voc_train.py --darknet ./darknet53-voc-result/darknet53_snapshot.npz -g 0 1
```

## 確認

```
python yolov3_voc_predict.py --yolo ./yolov3-voc-result/yolov3_final.npz --image ./data/image/dog.jpg --thresh 0.8
```
