# cycle_gan_inference

Original Model is from [junyanz_Cycle-Gan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix),I modified structure and applya another training strategy.

## To use this repo

The **only** three things you need to do :

1 Put sources image in the  `sources` directory.  

2 Select model in the test.py line 87
```
model_name = 'nature_photo.pth'  # e.g. nature_photo.pth
```

3 click `run` or press `shift+F10`. Then go to `res` directory to see the results.

## Notice that

All the image in the sources will be processed.

The results will be packaged in a directory under res. `Naming rules`: model_name_number.**e.g.** `nature_photo.pth_0`

Resolution of results are 768*768. There is many ways to get higher resolution(I am working on one based on a paper from 2022 CVPR)

## Lets see example results
**photo -> painting**  
<img src="/example_img/0.jpg" width = "400" height = "400" alt="" align=center />         <img src="/example_img/00.png" width = "400" height = "400" alt="" align=center />  
<img src="/example_img/1.jpg" width = "400" height = "400" alt="" align=center />         <img src="/example_img/11.png" width = "400" height = "400" alt="" align=center />
<img src="/example_img/2.jpg" width = "400" height = "400" alt="" align=center />         <img src="/example_img/22.png" width = "400" height = "400" alt="" align=center />
