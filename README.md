# UR10 robot with robotiq_2f_85 gripper

This is robotic env like gym FetchPickAndPlace-v1 but with UR10 robot arm and robotiq 2f 85 gripper.

# Generate URDF

install ros, [universal_robot](https://github.com/ros-industrial/universal_robot) and [robotiq](https://github.com/ros-industrial/robotiq)  
in catkin_ws do:
* ```source devel/setup.bash```
* ```rosrun xacro xacro ur10_robot_constr.urdf.xacro > my_robot.urdf```
* ```check_urdf my_robot.urdf```

# Final goal
Final goal was to train in with stable-baselines PPO2 to pick and place the cube to specific location. 
But it seems not feasible to do without hierarchical experience replay and inverse kinematics or fancy reward shaping or behaviour cloning methods. 