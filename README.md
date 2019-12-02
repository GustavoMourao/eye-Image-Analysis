### Features

Glaucoma image detection based on convolutional neural networks.

This repository has the structure:

- `Processor.py` - Class that implements some image filter signal processing techniques.

- `Interpreter.py` - Class that implements data augmentation procediment and CNN processing based on traditional approach and window optimization technique. 

- `test_processor.py` - Implements unit tests related to `Processor.py` class methods.

- `test_processor.py` - implements training proccess.

- `Graphs.py` - Class that implements graphs visualization.

### Refereces

##### Images Datasets

[1] Attila Budai, Joachim Hornegger, Georg Michelson: Multiscale Approach for Blood Vessel Segmentation on Retinal Fundus Images. In Invest Ophthalmol Vis Sci 2009;50: E-Abstract 325, 2009.

[Database Adress](https://www5.cs.fau.de/research/data/fundus-images/)


[2] C. Pena-Betancor, M. Gonzalez-Hernandez, F. Fumero-Batista, J. Sigut, E. Mesa, S. Alayon, and M. G. de la Rosa,
"Estimation of the relative amount of hemoglobin in the cup and neuro-retinal rim using stereoscopic color fundus images," IOVS, pp. IOVS–14–15592, Feb. 2015.

[Database Adress](http://medimrg.webs.ull.es/research/retinal-imaging/rim-one/)

[3] Zhang, Zhuo, et al. "Origa-light: An online retinal fundus image database for glaucoma analysis and research." 2010 Annual International Conference of the IEEE Engineering in Medicine and Biology. IEEE, 2010.

##### Window optimization

[4] Lee, Hyunkwang, Myeongchan Kim, and Synho Do. "Practical window setting optimization for medical image deep learning." arXiv preprint arXiv:1812.00572 (2018).

##### General

[5] Thomas Köhler, Attila Budai, Martin Kraus, Jan Odstrcilik, Georg Michelson, Joachim Hornegger. Automatic No-Reference Quality Assessment for Retinal Fundus Images Using Vessel Segmentation, 26th IEEE Internatioal Symposium on Computer-Based Medical Systems 2013, Porto
