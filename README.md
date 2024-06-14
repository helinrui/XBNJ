# 1、Easy and Precise Segmentation-Guided Diffusion Models

## 1) Train Your Own Models

### Data Preparation

Please put your training images in some dataset directory `DATA_FOLDER`, organized into train, validation and test split subdirectories. The images should be in a format that PIL can read (e.g. `.png`, `.jpg`, etc.). For example:

``` 
DATA_FOLDER
├── train
│   ├── tr_1.png
│   ├── tr_2.png
│   └── ...
├── val
│   ├── val_1.png
│   ├── val_2.png
│   └── ...
└── test
    ├── ts_1.png
    ├── ts_2.png
    └── ...
```

If you have segmentation masks, please put them in a similar directory structure in a separate folder `MASK_FOLDER`, with a subdirectory `all` that contains the split subfolders, as shown below. **Each segmentation mask should have the same filename as its corresponding image in `DATA_FOLDER`, and should be saved with integer values starting at zero for each object class, i.e., 0, 1, 2,...**.

If you don't want to train a segmentation-guided model, you can skip this step.

``` 
MASK_FOLDER
├── all
│   ├── train
│   │   ├── tr_1.png
│   │   ├── tr_2.png
│   │   └── ...
│   ├── val
│   │   ├── val_1.png
│   │   ├── val_2.png
│   │   └── ...
│   └── test
│       ├── ts_1.png
│       ├── ts_2.png
│       └── ...
```

### Training

The basic command for training a standard unconditional diffusion model is
```bash
CUDA_VISIBLE_DEVICES={DEVICES} python3 main.py \
    --mode train \
    --model_type DDIM \
    --img_size {IMAGE_SIZE} \
    --num_img_channels {NUM_IMAGE_CHANNELS} \
    --dataset {DATASET_NAME} \
    --img_dir {DATA_FOLDER} \
    --train_batch_size 16 \
    --eval_batch_size 8 \
    --num_epochs 400
```

where:
- `DEVICES` is a comma-separated list of GPU device indices to use (e.g. `0,1,2,3`).
- `IMAGE_SIZE` and `NUM_IMAGE_CHANNELS` respectively specify the size of the images to train on (e.g. `256`) and the number of channels (1 for greyscale, 3 for RGB).
- `model_type` specifies the type of diffusion model sampling algorithm to evaluate the model with, and can be `DDIM` or `DDPM`.
- `DATASET_NAME` is some name for your dataset (e.g. `breast_mri`).
- `DATA_FOLDER` is the path to your dataset directory, as outlined in the previous section.
- `--train_batch_size` and `--eval_batch_size` specify the batch sizes for training and evaluation, respectively. We use a train batch size of 16 for one 48 GB A6000 GPU for an image size of 256.
- `--num_epochs` specifies the number of epochs to train for (our default is 400).

#### Adding segmentation guidance, mask-ablated training, and other options

To train your model with mask guidance, simply add the options:
```bash
    --seg_dir {MASK_FOLDER} \
    --segmentation_guided \
    --num_segmentation_classes {N_SEGMENTATION_CLASSES} \
```

where:
- `MASK_FOLDER` is the path to your segmentation mask directory, as outlined in the previous section.
- `N_SEGMENTATION_CLASSES` is the number of classes in your segmentation masks, **including the background (0) class**.

To also train your model with mask ablation (randomly removing classes from the masks to each the model to condition on masks with missing classes; see our paper for details), simply also add the option `--use_ablated_segmentations`.

## 2) Evaluation/Sampling

Sampling images with a trained model is run similarly to training. For example, 100 samples from an unconditional model can be generated with the command:
```bash
CUDA_VISIBLE_DEVICES={DEVICES} python3 main.py \
    --mode eval_many \
    --model_type DDIM \
    --img_size 256 \
    --num_img_channels {NUM_IMAGE_CHANNELS} \
    --dataset {DATASET_NAME} \
    --eval_batch_size 8 \
    --eval_sample_size 100
```

Note that the code will automatically use the checkpoint from the training run, and will save the generated images to a directory called `samples` in the model's output directory. To sample from a model with segmentation guidance, simply add the options:
```bash
    --seg_dir {MASK_FOLDER} \
    --segmentation_guided \
    --num_segmentation_classes {N_SEGMENTATION_CLASSES} \
```
This will generate images conditioned on the segmentation masks in `MASK_FOLDER/all/test`. Segmentation masks should be saved as image files (e.g., `.png`) with integer values starting at zero for each object class, i.e., 0, 1, 2.


# 2、Rethinking Transfer Learning for Medical Image Classification(TTL)
## Train
### 2D experiment (XBNJ)
**block-wise TTL**

Please using following commands to train a model with federated learning strategy.
- **--model** specify model archicture: resnet50 | densenet201
- **--pretrained** specify source domain: imagenet | chexpert
- **--dataset** specify target dataset: XBNJ
- **--trunc** specify truncation point: {-1, 1, 2, 3}

```bash
python main.py --model resnet50 --bs 64 --data_parallel --num_workers 12 --max_epoch 200 --pretrained imagenet --dataset BIMCV --trunc -1 --exp 1 
```


**layer-wise TTL**

**--trunc** specify truncation point: {-1, 1, 2, ..., 16}

```bash
python main.py --model layerttl_resnet50 --bs 64 --data_parallel --num_workers 12 --max_epoch 200 --pretrained imagenet --dataset BIMCV --trunc -1 --exp 1 
```

### Test
**block-wise TTL**

```bash
python main.py --model resnet50 --bs 64 --data_parallel --num_workers 12 --max_epoch 200 --pretrained imagenet --dataset BIMCV --trunc -1 --exp 1 
```

**layer-wise TTL**

```bash
python main.py --model layerttl_resnet50 --bs 64 --data_parallel --num_workers 12 --max_epoch 200 --pretrained imagenet --dataset BIMCV --trunc -1 --exp 1 
```
