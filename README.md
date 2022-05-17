# survid
70 training images + data augmentation
2 pooling layers

|  Train | 6 without Gaussian  | 6 with Gaussian | 3 channel|
|--------|---------------------|-----------------|----------|
| loss   |     0.010412797     |   0.013968818   |0.010660506|
|accuracy|     0.9957977       |    0.99447954   | 0.9956917|
|  mIoU  |     0.9299962       |   0.9175082     | 0.9296004|

|validate| 6 without Gaussian  | 6 with Gaussian | 3 channel|
|--------|---------------------|-----------------|----------|
| loss   |     0.094205014     |  0.053493056    |0.059291173|
|accuracy|     0.97569734      |   0.98308015    |0.98359656|
|  mIoU  |    0.8325688        |   0.893182      |0.89593565|

![gaussian_train](https://user-images.githubusercontent.com/38833796/168426678-4bafb8c4-9edc-4b8f-a79f-a1d894957b93.png)
![gaussian_val](https://user-images.githubusercontent.com/38833796/168426677-a5d60093-f03b-44d1-b7da-aa367fd02555.png)

![3compare6_train](https://user-images.githubusercontent.com/38833796/168473038-cfac9929-61a3-48d2-9ed2-60bec7fc1e4b.png)
![3compare6_val](https://user-images.githubusercontent.com/38833796/168473030-dfd7cf81-8e19-459f-afca-af7be4cd73cd.png)


4 pooling layers
learning rate = 1e-5, learning rate = 1e-4 & learning rate = 1e-3
For learning rate = 1e-3
Train accuracy = 0.997998, Train mIoU = 0.9461413
Val accuracy = 0.9854351, Val mIoU = 0.90332425

![compare_rate_train](https://user-images.githubusercontent.com/38833796/168801176-17ff29e6-bebe-4216-9dd9-5dbb35113c9e.png)

![compare_rate](https://user-images.githubusercontent.com/38833796/168800974-07cfd70c-d30a-4805-805e-8518a48dd157.png)


