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

The results will be packaged in a directory under res. `Naming rules`: model_name_number.

Resolution of results are 768*768. There is many ways to get higher resolution(I am working on one based on a paper from 2022 CVPR)

## Lets see example results
![image](https://user-images.githubusercontent.com/89610539/179400998-0da09648-4ad2-496a-9c02-9dab9af4024c.png )=768*768
![image](https://user-images.githubusercontent.com/89610539/179401012-957b7ae8-f9b1-4050-b4c9-893bf93ca5dd.png )=768*768
