# UR10 robot with robotiq_2f_85 gripper

This is a robotic env like gym FetchPickAndPlace-v1 but with UR10 robot arm and robotiq 2f 85 gripper. It uses pybullet simulator to simulate physical interactions between robot and environment.   
![image](imgs/env_view.png)

# Generate URDF

To modify URDF model install ros, [universal_robot](https://github.com/ros-industrial/universal_robot) and [robotiq](https://github.com/ros-industrial/robotiq)  
in catkin_ws do:
* ```source devel/setup.bash```
* ```rosrun xacro xacro ur10_robot_constr.urdf.xacro > my_robot.urdf```
* ```check_urdf my_robot.urdf```

# Run
```python3 main.py```

# Final goal
Final goal was to train an RL agent using stable-baselines PPO2 algorithm to pick and place the cube to a specific location. 
But it seems not feasible to do without hierarchical experience replay and inverse kinematics or fancy reward shaping or behaviour cloning methods. 