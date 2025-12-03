## 依赖环境 (Requirements)
- Python ≥ 3.8
- OpenCV (`opencv-python`)
- NumPy
- Pillow
- tqdm
- grpcio
- protobuf
- lpips
- clean-fid


## 使用方法 (Usage)
```
conda create -n metric python=3.10
conda activate metric
pip install -r requirements.txt
cd scripts
sh run.sh
```