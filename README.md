### Features

Glaucoma image detection based on convolutional neural netoworks.

This repository has the structure:

- `Processor.py` - Class that implements some image filter signal processing techniques.

- `Interpreter.py` - Class that implements data augmentation procediment and CNN processing based on traditional approach and window optimization technique. 

- `test_processor.py` - Implements unit tests related to `Processor.py` class methods.

- `test_processor.py` - implements training proccess.


### Refereces

##### Images Dataset

[1] Attila Budai, Joachim Hornegger, Georg Michelson: Multiscale Approach for Blood Vessel Segmentation on Retinal Fundus Images. In Invest Ophthalmol Vis Sci 2009;50: E-Abstract 325, 2009.

##### Window optimization

[2] Lee, Hyunkwang, Myeongchan Kim, and Synho Do. "Practical window setting optimization for medical image deep learning." arXiv preprint arXiv:1812.00572 (2018).

[Database Adress](https://www5.cs.fau.de/research/data/fundus-images/)
