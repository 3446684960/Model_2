# 📖 Introduction

This competing approach consists of two sequential stages:

1. **Demosaicing** using **PPID**
2. **Pansharpening** using **PanGAN**

Notably, the demosaicing network does **not** require training.  
Only the pansharpening network (**PanGAN**) is trained in this framework.

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
### pansharpening
```
python train_pansharpening_simulate.py --idx 2 --data_path /data/wangn/DataSet --dataset CAVE --epochs 100 --train_size 64 --stride 32 --batch_size 16 --lr_decay --save_freq 5 --device 6 --noise_level 0.01 0.05
```

## 🔵 Chikusei
### pansharpening
```
python train_pansharpening_simulate.py --idx 4 --data_path /data/wangn/DataSet --dataset chikusei --epochs 1000 --train_size 64 --stride 32 --batch_size 16 --lr_decay --save_freq 5 --device 7 --noise_level 0.01 0.03
```

## 🟠 realworld dataset
### pansharpening
```
python train_pansharpening_realworld.py --idx 6 --data_path /data/wangn/DataSet --dataset real_world --epochs 50 --train_size 64 --stride 64 --batch_size 16 --lr_decay --save_freq 5 --visual --visual_freq 5 --device 0
```

# 2. Generate
## 🟢 CAVE
### generate all
```
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path /data/wangn/DataSet --dataset CAVE --load_ps_model ./2/model/best_40.pth --noise_level 0.01 0.05
```

### generate single
```
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path /data/wangn/DataSet --dataset CAVE --load_ps_model ./2/model/best_40.pth --noise_level 0.01 0.05 --data_id watercolors_ms.mat
```

## 🔵 Chikusei
### generate all
```
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path /data/wangn/DataSet --dataset chikusei --load_ps_model ./4/model/best_928.pth --noise_level 0.01 0.03
```

### generate single
```
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path /data/wangn/DataSet --dataset chikusei --load_ps_model ./4/model/best_928.pth --noise_level 0.01 0.03 --data_id 1.mat
```

## 🟠 RealWorld
### generate all
```
python generate.py --idx 1 --real_world --mosaic_save --pan_save --demosaic_save --data_path /data/wangn/DataSet --dataset real_world --load_ps_model ./6/model/best_43.pth
```

### generate single
```
python generate.py --idx 1 --real_world --mosaic_save --pan_save --demosaic_save --data_path /data/wangn/DataSet --dataset real_world --load_ps_model ./6/model/best_43.pth --data_id 2.raw 16.raw 31.raw
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
python test.py --idx 1 --simulate --data_path ./chikusei/1/result/mat/ --data_id 1.mat
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

## 🔵 ICVL
### visualize fused
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/fused/ --save_path ./ICVL/1/result/rgb/fused/ --data_type fused --data_id BGU_0522-1217.mat --detach --detach_size 128 128 --detach_coordinate 570 240 570 240 --boxcolor b --boxwidth 5
```

### visualize diffmap
```
python visualize.py --visual_task diffmap --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/fused/ --mat_path_for_diff ./ICVL/1/result/mat/gt/ --save_path ./ICVL/1/result/rgb/diffmap/ --data_id BGU_0522-1217.mat --detach --detach_size 128 128 --detach_coordinate 570 240 570 240 --boxcolor b --boxwidth 5 --mae_level 32 --sam_level 32
```

### visualize pan
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/pan/ --save_path ./ICVL/1/result/rgb/pan/ --data_type pan --data_id BGU_0522-1217.mat --detach --detach_size 128 128 --detach_coordinate 570 240 570 240 --boxcolor b --boxwidth 5
```

### visualize gt
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/gt/ --save_path ./ICVL/1/result/rgb/gt/ --data_type mosaic --data_id BGU_0522-1217.mat --detach --detach_size 128 128 --detach_coordinate 570 240 570 240 --boxcolor b --boxwidth 5
```

### visualize upmosaic
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/mosaic/ --save_path ./ICVL/1/result/rgb/upmosaic/ --data_type upmosaic --data_id BGU_0522-1217.mat --detach --detach_size 128 128 --detach_coordinate 570 240 570 240 --boxcolor b --boxwidth 5
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