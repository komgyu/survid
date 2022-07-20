# survid
Enviroment:
Anaconda 3
Python = 3.8
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
NumPy, etc....

The training data, validation data and test data are located in directory:./original_data, ./validate, ./test.
The images are scaled so there are ./scale files. The whole process will use the scaled images to fit the GPU size.

./others stored files for drawing plots and etc.
./model_records stored the history models.
./models stored the final models used for test.
./logs/Cross_records stored the history training data.
-------------------------------------------------------------------------------------------------------------------
For training, run main.py.
There are 2 kinds of input, 3-channel or 6-channel.
The default is 6 channel, if you want to switch to 3-channel:
modify line 46 in main.py: 6->3
modify line 172 in main.py: s->im
modify line 200 in main.py: s->im
modify line 229 in main.py: s->im

------------------------------------------------------------------------------------------------------------------
For test, run predict.py
The final models for 3 channel and 6 channel are located in ./models.
The default is also 6-channel input, if you want to switch to 3-channel:
modify line 100 in predict.py: snet_tmp6final.pt->snet_tmp3final.pt
modify line 100 in predict.py: 6->3
modify line 146 in predict.py: IF->image
