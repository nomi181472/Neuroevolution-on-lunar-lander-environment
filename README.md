
# Neuroevolution-on-lunar-lander-environment
<video width="640" height="360" controls>
  <source src="./videos/1.mp4" type="video/mp4">
  
</video>
Neuroevolution emerges as a compelling alternative to reinforcement learning, offering a gradient-free approach to optimize neural networks for skill discovery. It harnesses the power of evolutionary algorithms to iteratively refine network parameters, effectively navigating complex search spaces without relying on gradient information.


##1 Create Conda environment 
  E.g "conda create -n  neuro"

##2 activate conda env
  E.g "conda activate neuro"

##3 install neccessary packages
  conda install -c conda-forge gym box2d-py 
  pip install torch numpy tensorboard

##4 test using trained weights
  e.g python main.py

