A "light-weigth" Q-Table was given for testing purposes, since I couldn't upload a 9Gigabyte one directly. 
When the mp4 video is created using test.py, if it can't be visualized on local, this site can be used for visualization instead:	
	https://jumpshare.com/viewer/mp4
 The following need to be installed:
!pip install gymnasium[atari]
!pip install gymnasium[accept-rom-license]
!python -m atari_py.import_roms .\ROMS\ROMS
!pip install ale-py dill

The system also needs pytorch, moviepy to make the videos, tqdm, and matplotlib 
