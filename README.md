# siamese-network

* * * 
### config.py 
* data_dir : Dataset directory path. Must compose like {data_dir}/train, {data_dir}/val
* IMAGE_SIZE : Siamese network inference resolution (IMAGE_SIZE x IMAGE_SIZE)
* isColor : Siamese network inference channel
* min_lr, max_lr : cosine lr-scheduler min-max lr
* margin : Positive/Negative sample margin for calculate loss

* * * 

### model.py
* Network output channel = 30

* * *

### train.py 
* Training siamese network 
* During training, Visualize evaluation samples inference compressed by t-SNE from 30 channel to 2 channel
![siam](https://user-images.githubusercontent.com/59654033/208233386-26031f5f-3d6c-48e5-9a48-88fe0e0c3479.png)

* * * 

### predict.py
* Calculate mean of features from Train dataset path for pre-trained model
* Calculate features of predict source images
* Match features and save nearest object to 'pred_result/'
