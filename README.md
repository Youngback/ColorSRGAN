## Infrared image colorization based on an unbalanced dcgan ##

### Requirments
---------------------------------------
* nvidia gtx1080 11GB
* ubuntu 16.04 LTS
* python version = 3.xx
* backend: Tensorflow 1.7
* keras

### Model
---------------------------------------
#### GAN
![Alt text](./readme/model.png "GAN")

#### generator
![Alt text](./readme/model generator.png "GENERATOR")

#### discriminator
![Alt text](./readme/model discriminator.png "DISCRIMINATOR")

### Performance
---------------------------------------
* 64x64 to 256*256 size image

#### inference GPU
* nvidia gtx1080 11GB
* keras (backend tensorflow)

|time(msec)|(A)|(B)|(C)|(D)|
|---|:---:|---:|---:|---:|
|inference|5.035|7.127|4.110|5.260|

#### inference CPU
* intel core i7-7500U
(TODO)

### Result
---------------------------------------
#### NIR (near infrared) images
![Alt text](./readme/result nir.png "GAN")

#### FIR (far infrared) images
![Alt text](./readme/result fir.png "GAN")


