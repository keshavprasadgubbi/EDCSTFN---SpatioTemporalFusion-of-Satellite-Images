import pika
import json
import uuid
from pika import ConnectionParameters as param

class Singleton(object):
    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        return it
    def init(self, *args, **kwds):
        pass

class Queue(Singleton):
    def __init__(self, queue, host=param._DEFAULT, port=param._DEFAULT, username=None, password=None):
        if username is None or password is None:
            credentials = param._DEFAULT
        else:
            credentials = pika.credentials.PlainCredentials(username=username, password=password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port, credentials=credentials))
        self.channel = self.connection.channel()
        self.queue_name = queue
        self.channel.queue_declare(queue=queue)
        self.channel.confirm_delivery()
    
    def publish(self, data):
        self.channel.basic_publish(exchange='',
                                routing_key=self.queue_name,
                                body=json.dumps(data), mandatory=True)
    def close(self):
        self.connection.close()
                
       


