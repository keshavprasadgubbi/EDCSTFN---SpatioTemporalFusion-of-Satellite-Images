import psycopg2
import psycopg2.extras
import sqlite3

class DbConnection():

	def __init__(self, conf):
		self._isconnected = False
		self._is_sqlite = conf.get('dbfile') is not None
		self.cursor = None
		self.connection = self._create_connection(conf.get('host'), conf.get('user'), conf.get('password'), conf.get('db'), conf.get('port'), conf.get('dbfile'))
	

	def _create_connection(self, host=None, user=None, password=None, db=None, port=None, dbfile=None):
		try:
			if not self._isconnected:
				if self._is_sqlite:
					connection = sqlite3.connect(dbfile)
					self.cursor = connection.cursor()
				else:
					connection = psycopg2.connect(host=host, user=user, password=password, dbname=db, port=port)
					connection.autocommit = True
					self.cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
				self._isconnected = True
				return connection
		except Exception as e:
			print(e)
			raise Exception("Failed to create connection")

	def close(self):
		if self._isconnected:
			self.cursor.close()
			self.connection.close()

	def execute(self, query):
		try:
			if self._is_sqlite:
				return self.cursor.execute(query)
			else:
				return self.cursor.execute(query)
		except Exception as e:
			print(e)
	
	def executemany(self, query, args=None):
		try:
			if self._is_sqlite:
				return self.cursor.executemany(query, args)
			else:
				return psycopg2.extras.execute_values(self.cursor, query, args)
		except Exception as e:
			print(e)