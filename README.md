Opensource deep learning framework [TensorFlow](https://www.tensorflow.org) is used in **Facial Expression Recognition(FER)**. 
The trained models achieved 65% accuracy in fer2013. 

#### Dependencies

FER requires:
- Python (>= 3.3)
- TensorFlow (>= 1.1.0) [Installation](https://www.tensorflow.org/install/)
- OpenCV (python3-version) [Installation](http://docs.opencv.org/master/da/df6/tutorial_py_table_of_contents_setup.html)

Only tested in Ubuntu and macOS Sierra. Other platforms are not sure work well. When problems meet, open an issue, I'll do my best to solve that.

#### Usage
###### demo
The first stage demo can recognize your facial expression, the second stage demo adds voice recognition functions.

To run the first stage demo, just type:
```shell
python demo.py
```
Then the program will creat a window to display the scene capture by webcamera. You need press <kbd>SPACE</kbd> key to capture face in current frame and recognize the facial expression.

To run the second stage demo, just type:
```shell
python demo2.py
```
Then the program will creat a window to display the scene capture by webcamera. It will continually capture your facial expression and typing sounds, showing the results and give warm words to you.
