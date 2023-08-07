# reachability-aware-6dof-grasping
Implementation of our ICRA2020 paper "Learning to Generate 6-DoF Grasp Poses with Reachability Awareness"
*Lou, Xibai and Yang, Yang and Choi, Changhyun*<br/>
[arxiv.org/abs/1909.04840][1]<br/>




Motivated by the stringent requirements of unstructured real-world where a plethora of unknown objects reside in arbitrary locations of the surface, 
we propose a voxel-based deep 3D Convolutional Neural Network (3D CNN) that generates feasible 6-DoF grasp poses in unrestricted workspace with reachability awareness. 
Unlike the majority of works that predict if a proposed grasp pose within the restricted workspace will be successful solely based on grasp pose stability, 
our approach further learns a reachability predictor that evaluates if the grasp pose is reachable or not from robot's own experience. 
To avoid the laborious real training data collection, we exploit the power of simulation to train our networks on a large-scale synthetic dataset. 
This work is an early attempt that simultaneously learns grasping reachability while proposing feasible grasp poses with 3D CNN. 
Experimental results in both simulation and real-world demonstrate that our approach outperforms several other methods and achieves 82.5% grasping success rate on unknown objects.

If you find this code useful, please consider citing our work:

```
@inproceedings{lou2020learning,
  title={Learning to generate 6-dof grasp poses with reachability awareness},
  author={Lou, Xibai and Yang, Yang and Choi, Changhyun},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={1532--1538},
  year={2020},
  organization={IEEE}
}
```

## Dependencies
```
- Ubuntu 16.04
- Python 3
- PyTorch 0.4
```
We use [V-REP 3.5.0][2] as the simulation environment.

## Code

Start the V-REP simulation software first, and then open the scene file ```coppelia/scene.ttt``` to start the simulation environment. 
The pre-trained models is located at ```weights/gsp_pretrained.pt```


### Training
To train from scratch, run

```
python src/train.py
```

To collected your own data, specify your data output directory and run

```
python src/data_generation.py
```

### Testing
```
python src/test.py
```


[1]: https://arxiv.org/abs/1910.06404
[2]: http://coppeliarobotics.com/previousVersions
