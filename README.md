# parking_detect

Image classification task for parking spot monitoring using CCTV on *Tensorflow.*

### Example
![](parking_dataset_plain/parking.gif)

### Dataset of Choice

[__Car parking occupancy detection using smart camera networks and Deep Learning__](http://cnrpark.it/)
<br>
Giuseppe Amato, Fabio Carrara, Fabrizio Falchi, Claudio Gennaro and Claudio Vairo
<br>
___Accepted at [2016 IEEE Symposium on Computers and Communication (ISCC)](https://www.computer.org/csdl/proceedings/iscc/2016/12OmNviHKdN)___

### Training Setting

1. Classification on each parking spot ("0" for free and "1" for occupied), represented with bounding boxes
2. Each border was prepared manually by the dataset, 38k patches were utilized as training and test
3. Utilizing normalization as trainable parameter, but included in image preprocessing
4. Model Summary: *CNN* with Fully Connected Layers as classificator (4M trainable parameter)
5. *200px* * *200px* patch size


### Preparation
1. Clone this repository
```console
$ git@github.com:steven-ari/parking_detect.git
```
2. Download the dataset from [here](http://cnrpark.it/), download all four items listed under **"Dataset Download"**
3. Unzip content of .zip files according to the directory provided
4. Install requirements in requirements.txt
5. run *preprocessing/image_generator.py*: this creates training images out of predefined patch locations. <br/>Notice the date_name variable  
6. run *training/train_img_parking.py* to have a trained model  <br/>*training/eval_img_parking.py* for demo of prediction result

### Model Architecture (Simplified)
| Layer (type)    | Output Shape          | Param #   |
|  :---:          |     :---:             |  :---:    |
| BatchNorm       | (None, 200, 200, 3)   | 12        |
| Conv2D, Relu    | (None, 200, 200, 16)  | 448       |
| Conv2D, Relu    | (None, 100, 100, 32)  | 25120     |
| BatchNorm       | (None, 100, 100, 32)  | 25120     |
| Conv2D, Relu    | (None, 50, 50, 32)    | 9248      |
| Conv2D, Relu    | (None, 25, 25, 32)    | 9248      |
| dense (Dense)   | (None, 200)           | 461000    |
| Dropout         | (None, 200)           | 0         |
| dense (Dense)   | (None, 128)           | 25728     |
| dense (Dense)   | (None, 1)             | 129       |

Total params: 1,010,293 <br/>
Trainable params: 1,001,007 <br/>
Non-trainable params: 9,286

### Sources

[__Deep learning for decentralized parking lot occupancy detection__](http://cnrpark.it/)
<br>
Giuseppe Amato, Fabio Carrara, Fabrizio Falchi, Claudio Gennaro Carlo Meghini and Claudio Vairo
<br>
___Published on [Expert Systems with Applications 72, 327-334](https://www.sciencedirect.com/science/article/abs/pii/S095741741630598X)___

[__Car parking occupancy detection using smart camera networks and Deep Learning__](http://cnrpark.it/)
<br>
Giuseppe Amato, Fabio Carrara, Fabrizio Falchi, Claudio Gennaro and Claudio Vairo
<br>
___Accepted at [2016 IEEE Symposium on Computers and Communication (ISCC)](https://www.computer.org/csdl/proceedings/iscc/2016/12OmNviHKdN)___
