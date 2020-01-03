#!/usr/bin/env python
import smtplib
import socket
import fcntl
import struct
import time
from datetime import datetime
server = smtplib.SMTP('smtp.gmail.com', 587)

def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname[:15])
    )[20:24])

server.ehlo()
server.starttls()
server.ehlo()
server.login("Give your gmail","Nothingmatters1@")
x= datetime.now()
msg = get_ip_address('wlan0')			
msg1="it rockz..!!"
print(msg)		

server.sendmail("Give your gmail","repeat your gmail",msg1)
