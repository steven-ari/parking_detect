# parking_detect

### Dataset of Choice

[__Car parking occupancy detection using smart camera networks and Deep Learning__](http://cnrpark.it/)
<br>
Giuseppe Amato, Fabio Carrara, Fabrizio Falchi, Claudio Gennaro and Claudio Vairo
<br>
___Accepted at [2016 IEEE Symposium on Computers and Communication (ISCC)](https://www.computer.org/csdl/proceedings/iscc/2016/12OmNviHKdN)___

### Example
![](parking_dataset/parking.gif)

### Training Setting



### Preparation
1. Clone this repository
```console
$ git@github.com:steven-ari/parking_detect.git
```
2. Download the dataset from [here](http://cnrpark.it/), download all four items listed under **"Dataset Download"**
3. Install requirements in requirements.txt
4. run *preprocessing/image_generator.py*: this creates training images out of predefined patch locations. Notice the date_name variable  
5. run *training/train_img_parking.py* to have a trained model and *training/eval_img_parking.py* for demo of result

