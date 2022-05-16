# survid
70 training images + data augmentation
2 pooling layers

|  Train | 6 without Gaussian  | 6 with Gaussian | 3 channel|
|--------|---------------------|-----------------|----------|
| loss   |     0.010412797     |   0.013968818   |          |
|accuracy|     0.9957977       |    0.99447954   |          |
|  mIoU  |     0.9299962       |   0.9175082     |          |

|validate| 6 without Gaussian  | 6 with Gaussian | 3 channel|
|--------|---------------------|-----------------|----------|
| loss   |     0.094205014     |  0.053493056    |          |
|accuracy|     0.97569734      |   0.98308015    |          |
|  mIoU  |    0.8325688        |   0.893182      |          |

![gaussian_train](https://user-images.githubusercontent.com/38833796/168426678-4bafb8c4-9edc-4b8f-a79f-a1d894957b93.png)
![gaussian_val](https://user-images.githubusercontent.com/38833796/168426677-a5d60093-f03b-44d1-b7da-aa367fd02555.png)

![3compare6_train](https://user-images.githubusercontent.com/38833796/168473038-cfac9929-61a3-48d2-9ed2-60bec7fc1e4b.png)
![3compare6_val](https://user-images.githubusercontent.com/38833796/168473030-dfd7cf81-8e19-459f-afca-af7be4cd73cd.png)


4 pooling layers
learning rate = 1e-5
learning rate = 1e-4
![compare_rate](https://user-images.githubusercontent.com/38833796/168589468-0f809f3c-aaff-43ff-89fe-87d92fc99e44.png)
