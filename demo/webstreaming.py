from imutils.video import VideoStream, FPS
# from pyimagesearch.motion_detection.singlemotiondetector import SingleMotionDetector
from testforweb import real_timetest
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse  # 解析命令行参数的模块
import datetime
import imutils
import time
import cv2
from gevent import pywsgi

# initialize thw output frame and a lock used to ensure thread-safe
outputFrame = None
# 用于在更新outputFrame时确保线程安全行为，确保某个帧在更新时不被任何线程尝试读取
lock = threading.Lock()

# initialize the video stream and allow the camera sensor to warmup
app = Flask(__name__)

vs = VideoStream(src="rtsp://admin:l1234567@192.168.1.64:554/h265/ch1/main").start()  # src改为需要获取的网络摄像头的用户名：密码@ip
# vs = VideoStream(src=0).start()
time.sleep(2.0)


# 调用HTML文件
@app.route("/")
def index():
    return render_template("index.html")


# 循环视频流中的帧，应用检测函数，在outputFrame上绘制结果,并且保证能够并发
def detect_motion(frameCount):
    global vs, outputFrame, lock

    # 现在没有framecount帧就会计算累加权平均值
    # 达到了就会背景去除
    # md = SingleMotionDetector(accumWeight=0.1)
    total = 0
    # 循环遍历相机里的帧
    base = vs.read()
    base = imutils.resize(base, width=400)
    change_base = False
    fps = FPS().start()
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        if change_base:
            base = frame
            continue
        total = real_timetest(base,frame)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (7, 7), 0)

        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        #if total > frameCount:
         #   motion = md.detect(gray)
        # #    if motion is not None:
        #         (thresh, (minX, minY, maxX, maxY)) = motion
        #         cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 255), 2)
        # md.update(gray)
        # total += 1

        # 获取锁以确保在试图更新outputframe变量时不会意外读取
        fps.update()
        with lock:
            outputFrame = total.copy()
    fps.stop()
    print(fps.elapsed())  # 每帧视频的处理时间
    print(fps.fps())


# python生成器将outputframe编码未jpeg数据
def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            # 将frame编码为jpg减少负载 压缩失败就忽略这帧
            (flag, encodedImage) = cv2.imencode(".jpg",outputFrame)
            if not flag:
                continue
        # 将编码的JPEG以一个字节数组提供给一个可以解析它的网络浏览器
        yield(b'--frame\r\n'b'Content-Type:image/jpeg\r\n\r\n'+bytearray(encodedImage)+b'\r\n')

# 实时运动检测输出，通过generate函数编码为一个字节数组，浏览器将字节数组实时输出显示在浏览器中
@app.route("/video_feed")
def video_feed():
    return Response(generate(),mimetype="multipart/x-mixed-replace; boundary=frame")


# 解析命令行参数启动Flask
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--ip",type=str,default='0.0.0.0',
                    help="ip address of the device")
    ap.add_argument("-o","--port",type=int,default=8000,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f","--frame-count",type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    t = threading.Thread(target=detect_motion,args=(args["frame_count"],))
    t.daemon = True
    t.start()

    #start flask app
    server = pywsgi.WSGIServer((args["ip"],args["port"]),app)
    server.serve_forever()
    # app.run(host=args["ip"],port=args["port"],debug=True,
    #         threaded=True,use_reloader=False)
vs.stop()