# carnd-behavior-cloning

# Architecture
This project uses the architecture designed by NVIDIA's self-driving car team and documented in http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf.
The inputs consist of 66x200 RGB images, which are first normalized, then passed through 5 convolutional layers and 3 fully connected layers to finally produce a single float value representing the steering angle.
In order to prevent overfitting, a dropout layer of 0.5 is used after each convolutional layer.

# Approach to Problem
Before implementing the solution pipeline, some analysis work had been done to better understand the data and to prepare for debugging the pipeline and model later on.
First, the CSV file containing the Udacity-provided training data was read and organized in different lists with the same indexing for easier access. The data was then validated by counting the number of elements in each of the properties' array, and making sure they have the same length. A total of 8036 training records was found.
Then, 3 data records were chosen to be anaylized. One was a left turn, one was a snapshot of the car going straight, and the third was a right turn. The images and their corresponding steering angles (one negative, one zero, and one positive) were visualized, which made possible the design of the preprocessing algorithm: crop the top and bottom parts of the images, and resize to 66x200 to use them as inputs to the model.
Finally, using the 3 test images and their steering angles, the pipeline was developed and debugged, and a full model was trained.
The training is done through a generator, which select one index at random in the original training records, and generates 6 images with their corresponding steering angles: the left, center, and right camera images with steering corrections, as well as the symmetries of the 3 images with respect to the vertical axis. For the training itself, a total of 8000 x 6 = 48000 samples were used. Since the generator chooses an index at random, the 48000 samples (8000 indices) will not cover all training records, and will use some of them more than once. However, for model training purposes, this is a good enough approach. This also make sure that the data is shuffled between each of the 5 training epochs. For the training step, an Adam optimizer with mean squared error metric were used. Note that no validation set was created because the accuracy is not meaningful in this case, and the only real measurement of accuracy is testing in the simulator.

# Testing and Results
In order to test the model in the simulator, the drive.py script had to be slightly modified.
First, the same preprocessing steps are applied to the image from telemetry.
Then, the predicted steering angle was used, along with the current speed of the vehicle to adjust the throttle and steering angle of the car. Maximum and minimum speeds of 15 and 5 mi/hr were selected to balance between lap time and handling. Also, the throttle is controlled such that the car would slow down when the steering angle is large in either direction (a turn), and accelerate when the angle is small (straight line).
Using the model and pipeline described above, the car is able to run multiple laps in a row in the simulator.
