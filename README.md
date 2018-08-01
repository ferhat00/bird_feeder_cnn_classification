# bird_feeder_cnn_classification
A CNN used to classify images from a birdfeeder on the presence of birds.
This is a short exercise to classify whether birds are at your bird feeder using Keras, with Tensorflow as the backend. 

See my full article on medium: https://medium.com/@ferhat00/bird-classifier-for-bird-feeder-using-keras-a6c5762882c0

I will show an example of a self trained deep learning model which utilises 2D convolutional neural networks to predict whether the image is a bird or not a bird. Using an Nvidia GTX 1070 card, this takes about 10–15 minutes to train. The images generated from the system was handed over to me from the author of the project who hand sorted the images of birds versus no birds for the training set. An image of a bird at the bird feeder is shown below.

Sample webcam image:

![1_v45dwoe83x-imm6cuwqtxw](https://user-images.githubusercontent.com/30912225/43553816-d0a5d348-95e8-11e8-8ba7-074d13f6b287.jpeg)

The problem is that a lot of false positive images come about when there are no birds, and you see images of just the sky, or just from activating the light bulb in the room.

![1_1mdgnynd3-latswltmat8a](https://user-images.githubusercontent.com/30912225/43553912-4bf15fc2-95e9-11e8-904d-1a4b437faafb.jpeg)


Effect of augmentation:

![1_eroj3bxir4ljnaswb922cw](https://user-images.githubusercontent.com/30912225/43553795-b304189a-95e8-11e8-9484-6c9ffff005a0.png)

Summary of the CNN model used:

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 126, 126, 32)      896       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 63, 63, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 61, 61, 32)        9248      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 30, 30, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 28, 28, 64)        18496     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 14, 14, 64)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 12544)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                802880    
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 65        
=================================================================
Total params: 831,585
Trainable params: 831,585
Non-trainable params: 0
_________________________________________________________________

Training and validation accuracy:

![1_yezow5l24jnmutk_oez5xg](https://user-images.githubusercontent.com/30912225/43553833-e5018d28-95e8-11e8-97a3-81de17f7ec75.png)

Training and validation loss:

![1_yvpamzxvjkigxr37greaog](https://user-images.githubusercontent.com/30912225/43553837-eb15f3a2-95e8-11e8-969e-bc4321e47c4d.png)

This model is trained specifically to this birdfeeder, thus it overfits and achieves 96% accuracy to new test data of ~400 images from the same webcam and birdfeeder.
