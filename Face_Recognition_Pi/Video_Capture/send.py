import os
import paramiko

ssh = paramiko.SSHClient() 
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("184.73.42.128",username="kading",password="welcome1")
sftp = ssh.open_sftp()
sftp.put("/home/pi/Documents/Demo/videocapture/my_video.h264", "/home/kading/videocapture/my_video.h264")
sftp.close()
ssh.close()




