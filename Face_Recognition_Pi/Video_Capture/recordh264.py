import picamera
import time

camera = picamera.PiCamera()
camera.resolution = (1920, 1080)
camera.start_recording('my_video.h264')
print("Press Ctrl + C once you want to finish recording.")
while True:
    try:
	time.sleep(600)
    except KeyboardInterrupt:
	camera.stop_recording()
        break

