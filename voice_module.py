#!/usr/bin/env python

# Copyright (C) 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import print_function

import argparse
import os.path
import json
import time 
import requests
import google.oauth2.credentials
import datetime
from google.assistant.library import Assistant
from google.assistant.library.event import EventType
from google.assistant.library.file_helpers import existing_file

import google.assistant.library.assistant as gala

def process_event(event):
    """Pretty prints events.

    Prints all events that occur with two spaces between each new
    conversation and a single space between turns of a conversation.

    Args:
        event(event.Event): The current event to process.
    """
    if event.type == EventType.ON_CONVERSATION_TURN_STARTED:
        print()

    print(event)
    
    if event.type == EventType.ON_RECOGNIZING_SPEECH_FINISHED:
	speech_text = event.args["text"]
	print(" Going to send the speech text to the RPi: " + speech_text)	
	r=requests.post("http://192.168.1.96:80", data ={'direction': speech_text})
	print(" sent the speech text to the RPi: " + speech_text)

    if (event.type == EventType.ON_CONVERSATION_TURN_FINISHED and
            event.args and not event.args['with_follow_on_turn']):
        print()


def main():
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--credentials', type=existing_file,
                        metavar='OAUTH2_CREDENTIALS_FILE',
                        default=os.path.join(
                            os.path.expanduser('~/.config'),
                            'google-oauthlib-tool',
                            'credentials.json'
                        ),
                        help='Path to store and read OAuth2 credentials')
    args = parser.parse_args()
    with open(args.credentials, 'r') as f:
        credentials = google.oauth2.credentials.Credentials(token=None,
                                                            **json.load(f))
    print(credentials)
#    print(time.clock())
    cdt = datetime.datetime.now()
    print(cdt.minute)		
    with Assistant(credentials) as assistant:
#        assistant.set_mic_mute(True)
	print ('enter your pass code.. ;) ')
	while True:

		if(raw_input('enter another number..')=="100"):
			for event in assistant.start():
#				print("we r good?")
				assistant.start_conversation()	
#s				assistant.set_mic_mute(True)
#				assistant.start_conversation()
#				assistant.stop_conversation()
#				assistant.set_mic_mute(False)


                       						        
				#print(evt)		
				process_event(event)
#				assistant.start_conversation()
				
			break;
		elif (raw_input!="100"):
			print("nope nope!!")
			
#				
        


if __name__ == '__main__':
    main()
