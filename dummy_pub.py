#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import zmq
import time
from object_information import *
import json

def mogrify(topic, msg):
    """ json encode the message and prepend the topic """
    return bytes(topic + ' ' + json.dumps(msg), encoding='utf-8')

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5555")

#  Do 10 requests, waiting each time for a response
for request in range(10):
    print("Sending request %s …" % request)
    #socket.send_json({"object_info": [ObjectInformation(TYPE_CAR, STATUS_GO, 1, 2, 0).__dict__, ObjectInformation(TYPE_SEMI, STATUS_BREAK, 1, 10, 0).__dict__]})
    socket.send(mogrify('a', {"object_info": [ObjectInformation(TYPE_CAR, STATUS_GO, 1, 2, 0).__dict__, ObjectInformation(TYPE_SEMI, STATUS_BREAK, 1, 10, 0).__dict__]}))

    #  Do some 'work'
    time.sleep(1)