# 📖 Introduction

This competing approach consists of two sequential stages:

1. **Demosaicing** using **LSAN**
2. **Pansharpening** using **VBPN**

---

## 📂 Dataset Structure

The dataset is organized as follows:
```
Dataset/
├── CAVE/
│ ├── train/
│ └── test/
├── Chikusei/
│ ├── train/
│ └── test/
└── real_world/
├── train/
└── test/
```

## 🚀 Usage

The command lines for **training**, **generation**, **testing**, and **visualization** are provided below.

# 1. Train
## 🟢 CAVE 
### demosaic
```
python train_demosaic_simulate.py --idx 1 --data_path /data/wangn/DataSet --dataset CAVE --epochs 1000 --train_size 64 --stride 32 --batch_size 16 --lr_decay --save_freq 50 --noise_level 0.01 0.05
```

### pansharpening
```
python train_pansharpening_simulate.py --idx 2 --data_path /data/wangn/DataSet --dataset CAVE --epochs 20 --train_size 64 --stride 32 --batch_size 16  --resume_demosaic ./1/model/best_966.pth --lr_decay --save_freq 5 --device 2 --noise_level 0.01 0.05
```

## 🔵 Chikusei
### demosaic
```
python train_demosaic_simulate.py --idx 3 --data_path /data/wangn/DataSet --dataset chikusei --epochs 500 --train_size 64 --stride 32 --batch_size 16 --lr_decay --save_freq 50 --device 0 --noise_level 0.01 0.03
```

### pansharpening
```
python train_pansharpening_simulate.py --idx 4 --data_path /data/wangn/DataSet --dataset chikusei --epochs 10 --train_size 64 --stride 32 --batch_size 16  --resume_demosaic ./3/model/best_498.pth --lr_decay --save_freq 5 --device 6  --noise_level 0.01 0.03
```

## 🟠 realworld dataset
### demosaic
```
python train_demosaic_realworld.py --idx 5 --data_path /data/wangn/DataSet --dataset real_world --epochs 20 --train_size 64 --stride 32 --batch_size 16 --lr_decay --save_freq 1 --visual --visual_freq 1 --device 1
```

### pansharpening
```
python train_pansharpening_realworld.py --idx 6 --data_path /data/wangn/DataSet --dataset real_world --epochs 10 --train_size 64 --stride 64 --batch_size 16  --resume_demosaic ./5/model/best_17.pth --lr_decay --save_freq 1 --visual --visual_freq 1 --device 1
```

# 2. Generate
## 🟢 CAVE
### generate all
```
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path /data/wangn/DataSet --dataset CAVE --load_demosaic_model ./1/model/best_966.pth --load_ps_model ./2/model/best_2.pth --noise_level 0.01 0.05
```

### generate single
```
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path /data/wangn/DataSet --dataset CAVE --load_demosaic_model ./1/model/best_966.pth --load_ps_model ./2/model/best_2.pth --noise_level 0.01 0.05 --data_id watercolors_ms.mat
```

## 🔵 Chikusei
### generate all
```
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path /data/wangn/DataSet --dataset chikusei --load_demosaic_model ./3/model/best_498.pth --load_ps_model ./4/model/best_1.pth --noise_level 0.01 0.03
```

### generate single
```
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path /data/wangn/DataSet --dataset chikusei --load_demosaic_model ./3/model/best_498.pth --load_ps_model ./4/model/best_1.pth --noise_level 0.01 0.03 --data_id 3.mat
```

## 🟠 RealWorld
### generate all
```
python generate.py --idx 1 --real_world --mosaic_save --pan_save --demosaic_save --data_path /data/wangn/DataSet --dataset real_world --load_demosaic_model ./5/model/best_17.pth --load_ps_model ./6/model/best_3.pth
```

### generate single
```
python generate.py --idx 1 --real_world --mosaic_save --pan_save --demosaic_save --data_path /data/wangn/DataSet --dataset real_world --load_demosaic_model ./5/model/best_17.pth --load_ps_model ./6/model/best_3.pth --data_id 2.raw 16.raw 31.raw
```

# 3. test
## 🟢 Cave
### test all
```
python test.py --idx 1 --simulate --data_path ./CAVE/1/result/mat/
```

### test single
```
python test.py --idx 1 --simulate --data_path ./CAVE/1/result/mat/ --data_id watercolors_ms.mat
```
## 🔵 Chikusei
### test all
```
python test.py --idx 1 --simulate --data_path ./chikusei/1/result/mat/
```

### test single
```
python test.py --idx 1 --simulate --data_path ./chikusei/1/result/mat/ --data_id 3.mat
```
## 🟠 RealWorld
### test all
```
python test.py --idx 1 --real_world --data_path ./real_world/1/result/mat/ --patch_size 64
```

### test single
```
python test.py --idx 1 --real_world --data_path ./real_world/1/result/mat/ --patch_size 64 --data_id 2.mat 16.mat 31.mat
```

# 4. Visualize
## 🟢 CAVE
### visualize fused
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/fused/ --save_path ./CAVE/1/result/rgb/fused/ --data_type fused --data_id watercolors_ms.mat --detach --detach_size 50 50 --detach_coordinate 140 230 140 230 --boxcolor b --boxwidth 2
```

### visualize diffmap
```
python visualize.py --visual_task diffmap --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/fused/ --mat_path_for_diff ./CAVE/1/result/mat/gt/ --save_path ./CAVE/1/result/rgb/diffmap/ --data_id watercolors_ms.mat --detach --detach_size 50 50 --detach_coordinate 140 230 140 230 --boxcolor b --boxwidth 2 --mae_level 64 --sam_level 64
```

### visualize pan
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/pan/ --save_path ./CAVE/1/result/rgb/pan/ --data_type pan --data_id watercolors_ms.mat --detach --detach_size 50 50 --detach_coordinate 140 230 140 230 --boxcolor b --boxwidth 2
```

### visualize gt
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/gt/ --save_path ./CAVE/1/result/rgb/gt/ --data_type mosaic --data_id watercolors_ms.mat --detach --detach_size 50 50 --detach_coordinate 140 230 140 230 --boxcolor b --boxwidth 2
```

### visualize upmosaic
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/mosaic/ --save_path ./CAVE/1/result/rgb/upmosaic/ --data_type upmosaic --data_id watercolors_ms.mat --detach --detach_size 50 50 --detach_coordinate 140 230 140 230 --boxcolor b --boxwidth 2
```

## 🔵 Chikusei
### visualize fused
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./chikusei/1/result/mat/fused/ --save_path ./chikusei/1/result/rgb/fused/ --data_type fused --data_id 3.mat --detach --detach_size 40 40 --detach_coordinate 80 240 80 240 --boxcolor b --boxwidth 2
```

### visualize diffmap
```
python visualize.py --visual_task diffmap --spatial_ratio 8 --mat_path ./chikusei/1/result/mat/fused/ --mat_path_for_diff ./chikusei/1/result/mat/gt/ --save_path ./chikusei/1/result/rgb/diffmap/ --data_id 3.mat --detach --detach_size 40 40 --detach_coordinate 80 240 80 240 --boxcolor b --boxwidth 2 --mae_level 32 --sam_level 32
```

### visualize pan
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./chikusei/1/result/mat/pan/ --save_path ./chikusei/1/result/rgb/pan/ --data_type pan --data_id 3.mat --detach --detach_size 40 40 --detach_coordinate 80 240 80 240 --boxcolor b --boxwidth 2
```

### visualize gt
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./chikusei/1/result/mat/gt/ --save_path ./chikusei/1/result/rgb/gt/ --data_type mosaic --data_id 3.mat --detach --detach_size 40 40 --detach_coordinate 80 240 80 240 --boxcolor b --boxwidth 2
```

### visualize upmosaic
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./chikusei/1/result/mat/mosaic/ --save_path ./chikusei/1/result/rgb/upmosaic/ --data_type upmosaic --data_id 3.mat --detach --detach_size 40 40 --detach_coordinate 80 240 80 240 --boxcolor b --boxwidth 2
```

## 🟠 real-world
### visualize fused
#### img1
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/fused/ --save_path ./real_world/1/result/rgb/fused/ --data_id 2.mat --data_type fused --detach --detach_size 192 192 --detach_coordinate 860 510 860 510 --boxwidth 8 --boxcolor b
```
#### img2
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/fused/ --save_path ./real_world/1/result/rgb/fused/ --data_id 16.mat --data_type fused --detach --detach_size 192 192 --detach_coordinate 750 1170 750 1170 --boxwidth 8 --boxcolor b
```
#### img3
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/fused/ --save_path ./real_world/1/result/rgb/fused/ --data_id 31.mat --data_type fused --detach --detach_size 192 192 --detach_coordinate 460 1170 460 1170 --boxwidth 8 --boxcolor b
```

### visualize pan
#### img1
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/pan/ --save_path ./real_world/1/result/rgb/pan/ --data_id 2.mat --data_type pan --detach --detach_size 192 192 --detach_coordinate 860 510 860 510 --boxwidth 8 --boxcolor b
```
#### img2
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/pan/ --save_path ./real_world/1/result/rgb/pan/ --data_id 16.mat --data_type pan --detach --detach_size 192 192 --detach_coordinate 750 1170 750 1170 --boxwidth 8 --boxcolor b
```
#### img3
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/pan/ --save_path ./real_world/1/result/rgb/pan/ --data_id 31.mat --data_type pan --detach --detach_size 192 192 --detach_coordinate 460 1170 460 1170 --boxwidth 8 --boxcolor b
```

### visualize upmosaic
#### img1
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/mosaic/ --save_path ./real_world/1/result/rgb/upmosaic/ --data_id 2.mat --data_type upmosaic --detach --detach_size 192 192 --detach_coordinate 860 510 860 510 --boxwidth 8 --boxcolor b
```
#### img2
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/mosaic/ --save_path ./real_world/1/result/rgb/upmosaic/ --data_id 16.mat --data_type upmosaic --detach_size 192 192 --detach --detach_coordinate 750 1170 750 1170 --boxwidth 8 --boxcolor b
```
#### img3
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/mosaic/ --save_path ./real_world/1/result/rgb/upmosaic/ --data_id 31.mat --data_type upmosaic --detach --detach_size 192 192 --detach_coordinate 460 1170 460 1170 --boxwidth 8 --boxcolor b
```