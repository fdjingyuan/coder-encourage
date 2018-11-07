# /usr/bin/python3
import cv2
import numpy as np
import sys
import tensorflow as tf
import random
import time
import random
import traceback
from model import predict, image_to_tensor, deepnn

CASC_PATH = './data/haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']


def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    # None is no face found in image
    if not len(faces) > 0:
        return None, None
    max_are_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
            max_are_face = face
    # face to image
    face_coor = max_are_face
    image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
    # Resize image to network size
    try:
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
    except Exception:
        print("[+} Problem during resize")
        return None, None
    return image, face_coor


def face_dect(image):
    """
    Detecting faces in image
    :param image: 
    :return:  the coordinate of max face
    """
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    if not len(faces) > 0:
        return None
    max_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_face[2] * max_face[3]:
            max_face = face
    face_image = image[max_face[1]:(max_face[1] + max_face[2]), max_face[0]:(max_face[0] + max_face[3])]
    try:
        image = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+} Problem during resize")
        return None
    return face_image


def resize_image(image, size):
    try:
        image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("+} Problem during resize")
        return None
    return image


def draw_emotion():
    pass

modelPath = './ckpt/'
showBox = True
tf.reset_default_graph()

face_x = tf.placeholder(tf.float32, [None, 2304])
y_conv = deepnn(face_x)
probs = tf.nn.softmax(y_conv)

saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(modelPath)
sess = tf.Session()
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Restore model sucsses!!\nNOTE: Press SPACE on keyboard to capture face.')



# 'angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral'
# emoji_list和text_list结构差不多，都是一个map
# label -> list，从预测的类别号到一个list，显示的时候会随机从list里面挑一个显示

emoji_list = {}
for index, emotion in enumerate(EMOTIONS):
    tmp = []
    for i in range(1, 4):
        tmp.append(cv2.imread('./data/emojis/' + emotion + str(i) + '.jpg', -1))
    emoji_list[index] = tmp

text_list = {
     0 : [
        'It\'s ok. You can do it! Do not be angry',
        'You can also think with being angry',
        'Everyone gets angry sometimes.'
    ],
    1: [
        'Hold on please! Do not be disgusted',
        'Everyone will falied. Never mind!',
        'You are not debugging alone.'
    ],
    2: [
        'Don\'t be afraid, you will do it!'
        'Fear is useless, be brave please.'
        'There is nothing to worry. All will be fine.'
        'Bug is nothing!'
    ],
    3: [
        'Seems like coding can sometimes be fun.',
        'Happy coding.',
        'While many coders are still in sadness, glad you are not like them.',
        'Finished another feature?',
        'Good job!'
    ],
    4: [
        'Cheer up, your program will finally work one day.',
        'You haven\'t type True as Ture have you?',
        'Cheer up! Let it go!',
        'Never mind. Have a break.'
    ],
    5: [
        'What is the problem? You can find it',
        'Astonished? You can fix the bug.'
    ],
    6: [
        'Come on! Smile a little~'
    ]
}


audio_text = {
    'angry':[
        'Your typing sound seems angry...take it easy~'
    ],
    'normal':[
        'You are coding fluently, right? Keep going!'
    ],
    'thinking':[
        'I guess you are thinking now. You will make it! '
    ]
}


def random_choose(list_):
    '''
    随机从列表里面挑一个
    '''
    i = random.randint(0, len(list_) - 1)
    return list_[i]

class Parser(object):
    
    def __init__(self, emoji_list, text_list, showBox=True, parseEverySecond=1):
        '''
        emoji_list: emoji的label->list映射
        text_list: text的label->list映射
        showBox：是否显示人脸框
        parseEverySecond：每多少秒处理一次图片。这个值越小，图片文字变化速度越快，同时对性能要求就越高
        '''
        self.showBox = True
        self.previous_second = 0
        self.emoji_list = emoji_list
        self.text_list = text_list
        self.text_to_show = None
        self.emoji_to_show = None
        self.parseEverySecond = parseEverySecond
    
    def parse_face(self, frame):
        '''
        检测人脸。如果showBox = True就标注人脸。返回检测到的人脸区域。
        '''
        detected_face, face_coor = format_image(frame)
        if self.showBox:
            if face_coor is not None:
                [x, y, w, h] = face_coor
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        if face_coor is None:
            return None
        else:
            return detected_face
    
    def predict_emotion(self, detected_face):
        '''
        预测表情，返回一个形状为(7, )的向量
        '''
        tensor = image_to_tensor(detected_face)
        result = sess.run(probs, feed_dict={face_x: tensor})
        return result
    
    def draw_emoji(self, frame):
        '''
        画 self.emoji_to_show 里面的表情
        '''
        if self.emoji_to_show is not None:
            for c in range(0, 3):
                frame[200:320, 10:130, c] = self.emoji_to_show[:, :, c] * (self.emoji_to_show[:, :, c] / 255.0) + \
                    frame[200:320, 10:130, c] * (1.0 - self.emoji_to_show[:, :, c] / 255.0)
    
    def draw_text(self, frame):
        '''
        画 self.text_to_show 里面的文字
        '''
        if self.text_to_show is not None:
            cv2.putText(frame, self.text_to_show, (10, 350), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)

    def parse(self, frame):
        '''
        主处理函数
        '''
        detected_face = self.parse_face(frame)
        # 如果和上一次处理间隔大于self.parseEverySecond秒，就进行处理
        if time.time() - self.previous_second > self.parseEverySecond:
            # 如果有人脸则调用情绪分类模型
            if detected_face is not None:
                emotion_prob = self.predict_emotion(detected_face)
                prob_index = np.argmax(emotion_prob[0])
                # 随机选emoji和text
                self.emoji_to_show = random_choose(self.emoji_list[prob_index])
                self.text_to_show = random_choose(self.text_list[prob_index])
            # 记录当前处理时间
            self.previous_second = time.time()
        
        # 画出emoji和text
        self.draw_emoji(frame)
        self.draw_text(frame)

video_captor = cv2.VideoCapture(0)
parser = Parser(emoji_list, text_list, parseEverySecond=2)

try:
    from audio_rec import AudioRec
    rec_thread = AudioRec()
    rec_thread.start()
    while True:
        # ret, frame = video_captor.read()
        import numpy as np
        frame = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        # 处理图片
        parser.parse(frame)
        #处理声音
        samples = None
        image = None
        rec_thread.lock.acquire()
        if rec_thread.samples is not None:
            samples = rec_thread.samples.copy()
        if rec_thread.image is not None:
            image = rec_thread.image.copy()
            print(rec_thread.image.shape)
        rec_thread.lock.release()
        # 显示图片
        cv2.imshow('face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    traceback.print_exc()
finally:
    video_captor.release()
    rec_thread.stop()