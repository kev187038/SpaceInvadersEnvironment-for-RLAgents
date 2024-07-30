In this project the SpaceInvaders gaming Environment was used to train and test a Reinforcement Learning agent, using different techniques. 
In the repo are available the source code, the test results, and a report with all the details. The experience was important to learn how to implement and neural network architecture and both basic and
Q-Learning. 


A "light-weigth" Q-Table was given for testing purposes, since I couldn't upload a 9Gigabyte one directly. The bigger one can be trained and tested on the provided Colab Notebook.
When the mp4 video is created using test.py, if it can't be visualized on local, this site can be used for visualization instead:	
	https://jumpshare.com/viewer/mp4
 The following need to be installed:
!pip install gymnasium[atari]
!pip install gymnasium[accept-rom-license]
!python -m atari_py.import_roms .\ROMS\ROMS
!pip install ale-py dill

The system also needs pytorch, moviepy to make the videos, tqdm, and matplotlib 

Colab Link: https://colab.research.google.com/drive/1XHU2VgTdAUPu5-0_FEaPgSjlsAPiNv6R?usp=sharing

