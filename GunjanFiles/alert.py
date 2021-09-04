import os
import requests

def send_alert(message):
    webhook = os.environ.get('ALERT_URL')
    if webhook is not None:
        req = requests.post(webhook, headers={'Content-Type': 'application/json; charset=UTF-8'}, json={'text': message})
        return req
    print("No ALERT_URL set in the environment")