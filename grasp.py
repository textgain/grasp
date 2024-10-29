# encoding: utf-8

##### GRASP.PY ####################################################################################

__version__   =  '3.2.3'
__license__   =  'BSD'
__credits__   = ['Tom De Smedt', 'Guy De Pauw', 'Walter Daelemans']
__email__     =  'info@textgain.com'
__author__    =  'Textgain'
__copyright__ =  'Textgain'

###################################################################################################

# Copyright 2023 Textgain
#
# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors 
#    may be used  to endorse or promote products derived from this software 
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR 
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND 
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY 
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

###################################################################################################
# Grasp.py is a collection of simple algorithms, functions and classes for data mining & analytics:

# WWW  Web Mining                   search engines, servers, HTML DOM + CSS selectors, plaintext
# DB   Databases                    comma-separated values, dates, SQL
# NLP  Natural Language Processing  tokenization, part-of-speech tagging, sentiment analysis
# ML   Machine Learning             clustering, classification, confusion matrix, n-grams
# NET  Network Analysis             shortest paths, centrality, components, communities
# ETC                               recipes for functions, strings, lists, dates, ...

# Grasp.py is based on the Pattern toolkit (https://github.com/clips/pattern), focusing on brevity.
# Most functions have around 10 lines of code, and most algorithms have around 25-50 lines of code.
# Most classes have about 50-75 lines of code.
###################################################################################################

import sys
import os
import io
import re
import platform
import inspect
import logging
import traceback
import threading
import subprocess
import multiprocessing
import multiprocessing.pool
import queue
import itertools
import collections
import unicodedata
import string
import socket; socket.setdefaulttimeout(10)
import wsgiref
import wsgiref.simple_server
import urllib
import smtplib
import hashlib
import hmac
import base64
import binascii
import email
import xml.etree.ElementTree as ElementTree
import sqlite3 as sqlite
import csv as csvlib
import json
import zipfile
import tempfile
import mimetypes
import glob
import time
import datetime
import random
import math
import heapq
import bisect

OS  = platform.system()

PY2 = sys.version.startswith('2')
PY3 = sys.version.startswith('3')

if PY3:
    unicode, basestring = str, str

if PY3:
    import collections.abc
else:
    collections.abc = collections

if PY3:
    from html.parser import HTMLParser
    from html import unescape
else:
    from HTMLParser import HTMLParser
    unescape = HTMLParser().unescape

if PY3:
    import http.server as BaseHTTPServer
    import socketserver as SocketServer
else:
    import BaseHTTPServer
    import SocketServer

if PY3:
    import http.cookiejar as cookielib
else:
    import cookielib

if PY3:
    import urllib.request
    import urllib.parse
else:
    import urllib2
    import urlparse

    urllib.error           = urllib2
    urllib.request         = urllib2
    urllib.parse           = urlparse
    urllib.parse.urlencode = urllib.urlencode
    urllib.parse.unquote   = urllib.unquote
    urllib.parse.quote     = urllib.quote

# In Python 2, Class.__str__ returns a byte string.
# In Python 3, Class.__str__ returns a Unicode string.

# @printable
# class X(object):
#     def __str__(self):
#         return unicode(' ')

# works on both Python 2 & 3.

def printable(cls):
    """ @printable class defines class.__unicode__ in Python 2.
    """
    if PY2:
        if hasattr(cls, '__str__'):
            cls.__unicode__ = cls.__str__
            cls.__str__ = lambda self: self.__unicode__().encode('utf-8')
    return cls

REGEX = type(re.compile(''))

# isinstance(re.compile(''), REGEX)

###################################################################################################

#---- STATIC --------------------------------------------------------------------------------------

def static(**kwargs):
    """ The @static() decorator initializes static variables.
    """
    def decorator(f):
        for k, v in kwargs.items():
            setattr(f, k, v)
        return f
    return decorator

# @static(i=0)
# def uid():
#     uid.i += 1
#     return uid.i

#---- CACHED --------------------------------------------------------------------------------------

def cached(f):
    """ The @cached() decorator stores function return values.
    """
    v = {}
    def decorator(*args, **kwargs):
        k = repr((args, kwargs))
        if not k in v:
            v[k] = f(*args, **kwargs)
        return v[k]
    return decorator

# @cached
# def fib(n):
#     return n if n < 2 else fib(n-1) + fib(n-2)

#---- PARALLEL ------------------------------------------------------------------------------------
# Parallel processing uses multiple CPU's to execute multiple processes simultaneously.

def parallel(f, values=[], *args, **kwargs):
    """ Returns an iterator of f(v, *args, **kwargs)
        for values=[v1, v2, ...], using available CPU's.
    """
    p = multiprocessing.Pool(processes=None)
    p = p.imap(_worker, ((f, v, args, kwargs) for v in values))
    return p

def _worker(x):
    f, v, args, kwargs = x
    return f(v, *args, **kwargs)

# for v in parallel(pow, (1, 2, 3), 2):
#     print(v)

#---- ASYNC ---------------------------------------------------------------------------------------
# Asynchronous functions are executed in a separate thread and notify a callback function 
# (instead of blocking the main thread).

def asynchronous(f, callback=lambda v, e: None, blocking=False):
    """ Returns a new function that calls 
        callback(value, exception=None) when done.
    """
    def thread(*args, **kwargs):
        def worker(callback, f, *args, **kwargs):
            try:
                v = f(*args, **kwargs)
            except Exception as e:
                callback(None, e)
            else:
                callback(v, None)
        t = threading.Thread
        t = t(target=worker, args=(callback, f) + args, kwargs=kwargs)
        t.daemon = not blocking
        t.start()
        return t
    return thread

asynch = asynchronous

# def ping(v, e=None):
#     if e: 
#         raise e
#     print(v)
# 
# pow = asynchronous(pow, ping)
# pow(2, 2)
# pow(2, 3) #.join(1)
# 
# for _ in range(10):
#     time.sleep(0.1)
#     print('...')

# Atomic operations are thread-safe, e.g., dict.get() or list.append(),
# but not all operations are atomic, e.g., dict[k] += 1 needs a lock.

Lock = threading.RLock
lock = threading.RLock()

def atomic(f):
    """ The @atomic decorator executes a function thread-safe.
    """
    def decorator(*args, **kwargs):
        with lock:
            return f(*args, **kwargs)
    return decorator

# hits = collections.Counter()
# 
# @atomic
# def hit(k):
#     hits[k] += 1

MINUTE, HOUR, DAY = 60, 60*60, 60*60*24

def scheduled(interval=MINUTE, blocking=False):
    """ The @scheduled decorator executes a function periodically (async).
    """
    def decorator(f):
        def timer():
            while 1:
                time.sleep(interval)
                try:
                    f()
                except: # print traceback & keep scheduling:
                    sys.stderr.write(traceback.format_exc())
        t = threading.Thread(target=timer)
        t.daemon = not blocking
        t.start()
        return f
    return decorator

# @scheduled(1)
# @atomic
# def update():
#     print('updating...')

def retry(exception, tries, f, *args, **kwargs):
    """ Returns the value of f(*args, **kwargs).
        Retries if the given exception is raised.
    """
    for i in range(tries + 1):
        try:
            return f(*args, **kwargs)
        except exception as e:
            if i < tries: 
                time.sleep(2 ** i) # exponential backoff (1, 2, 4, ...)
        except Exception as e:
            raise e
    raise e

# def search(q,n):
#     print('searching %s' % q)
#     raise ValueError
# 
# retry(ValueError, 3, search, 'cats')

# Asynchronous + retry:
# f = asynchronous(lambda x: retry(Exception, 2, addx, x), callback)

###################################################################################################

#---- LAZY ----------------------------------------------------------------------------------------
# A lazy container takes lambda functions as values, which are evaluated when retrieved.

class LazyDict(collections.abc.MutableMapping):

    def __init__(self, *args, **kwargs):
        self._dict = dict(*args, **kwargs)
        self._done = set()

    def __setitem__(self, k, v):
        self._dict[k] = v

    def __getitem__(self, k):
        v = self._dict[k]
        if not k in self._done:
            self._dict[k] = v = v()
            self._done.add(k)
        return v

    def __delitem__(self, k):
        self._dict.pop(k)
        self._done.remove(k)

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def __repr__(self):
        return repr(dict(self))

# models = LazyDict()
# models['en'] = lambda: Perceptron('large.json')

#---- PERSISTENT ----------------------------------------------------------------------------------
# A persistent container stores values in a file.

class PersistentDict(dict):

    def __init__(self, path, *args, **kwargs):
        if os.path.exists(path or ''):
            self.update(json.load(open(path, 'rb')))
        self.update(*args)
        self.update(kwargs)
        self.path = path

    @atomic # (thread-safe)
    def save(self, path=''):
        json.dump(self, open(path or self.path, 'w'), ensure_ascii=False, indent=4) # JSON

# db = PersistentDict('db.json', {'k': 'v'})
# db.save()

###################################################################################################

#---- LOG -----------------------------------------------------------------------------------------
# Functions that access the internet must report the visited URL using the standard logging module.
# See also: https://docs.python.org/2/library/logging.html

SIGNED = '%(time)s %(file)s:%(line)s %(function)s: %(message)s\n' # 12:59:59 grasp.py:1000 <module>

log = logging.getLogger(__name__)
log.level = logging.DEBUG

if not log.handlers:
    log.handlers.append(logging.NullHandler())

class Log(collections.deque, logging.Handler):

    def __init__(self, n=100, file=None, format=SIGNED, date='%Y-%m-%d %H:%M:%S'):
        """ A list of n latest log messages, optionally with a file-like back-end.
        """
        collections.deque.__init__(self, maxlen=n)
        logging.Handler.__init__(self)
        log.handlers.append(self)

        self.file   = file
        self.format = format
        self.date   = date

    def emit(self, r):
        r = {                                       #  log.info('test')
            'time' : r.created + r.relativeCreated, #  date().timestamp
            'type' : r.levelname.lower(),           # 'info'
         'message' : r.getMessage(),                # 'test'
        'function' : r.funcName,                    # '<module>'
          'module' : r.module,                      # 'grasp'
            'path' : r.pathname,                    # 'grasp.py'
            'file' : r.filename,                    # 'grasp.py'
            'line' : r.lineno,                      #  1234
        }
        if self.file:
            self.file.write(
                self.format % dict(r, 
                    time=datetime.datetime.fromtimestamp(r['time']).strftime(self.date)))
        self.append(r)
        self.update(r)

    def update(self, event):
        pass

    def __del__(self):
        try:
            log.handlers.remove(self)
        except:
            pass

def debug(file=sys.stdout, format=SIGNED, date='%Y-%m-%d %H:%M:%S'):
    debug.log = Log(0, file, format, date)

# debug()
# debug(open(cd('log.txt'), 'a'))
# request('https://textgain.com')

###################################################################################################

#---- UNICODE -------------------------------------------------------------------------------------
# The u() function returns a Unicode string (Python 2 & 3).
# The b() function returns a byte string, encoded as UTF-8.

# We use u() as early as possible on all input (e.g. HTML).
# We use b() on URLs.

def u(v, encoding='utf-8'):
    """ Returns the given value as a Unicode string.
    """
    if PY3 and isinstance(v, bytes) or \
       PY2 and isinstance(v, str):
        for e in ((encoding,), ('windows-1252',), ('utf-8', 'ignore')):
            try:
                return v.decode(*e)
            except:
                pass
        return v
    if isinstance(v, unicode):
        return v
    return (u'%s' % v) # int, float

def b(v, encoding='utf-8'):
    """ Returns the given value as a byte string.
    """
    if isinstance(v, unicode):
        for e in ((encoding,), ('windows-1252',), ('utf-8', 'ignore')):
            try:
                return v.encode(*e)
            except:
                pass
        return v
    if PY3 and isinstance(v, bytes):
        return v
    if PY2 and isinstance(v, str):
        return v
    return (u'%s' % v).encode()

#---- ITERATION -----------------------------------------------------------------------------------

def first(n, a):
    """ Returns a iterator of values from index 0 to n.
    """
    return iter(itertools.islice(a, 0, n))

def sliced(a, *ijn):
    """ Returns an iterator of values from index i to j, by step n.
    """
    return iter(itertools.islice(a, *ijn))

def shuffled(a):
    """ Returns an iterator of values in the list, in random order.
    """
    a = list(a)
    random.shuffle(a)
    return iter(a)

def unique(a):
    """ Returns an iterator of unique values in the list, in order.
    """
    s = set() # seen?
    return iter(v for v in a if not (v in s or s.add(v)))

def chunks(a, n=2):
    """ Returns an iterator of tuples of n consecutive values.
    """
    a = list(a)
    return iter(zip(*(a[i::n] for i in range(n))))

# for v in chunks([1, 2, 3, 4], n=2): # (1, 2), (3, 4)
#     print(v)

def nwise(a, n=2):
    """ Returns an iterator of tuples of n consecutive values (rolling).
    """
    a = itertools.tee(a, n)
    a =(itertools.islice(a, i, None) for i, a in enumerate(a))
    a = zip(*a)
    a = iter(a)
    return a

# for v in nwise([1, 2, 3, 4], n=2): # (1, 2), (2, 3), (3, 4)
#     print(v)

def flatten(a, type=list):
    """ Returns a list of nested values (depth-first).
    """
    q = []
    for v in a:
        if isinstance(v, type):
            q.extend(flatten(v, type))
        else:
            q.append(v)
    return q

# print(flatten([1, [2, [3, 4]]]))

def group(a, key=lambda v: v):
    """ Returns a dict of lists of values grouped by key(v).
    """
    g = {}
    for v in a:
        g.setdefault(key(v), []).append(v)
    return g

# print(group([1, 1, 1, 2, 2, 3]))

#---- RANDOM --------------------------------------------------------------------------------------

def choices(a, weights=[], k=1):
    """ Returns a list of k random elements from the given list,
        with optional (non-negative) probabilities.
    """
    if weights:
        n = 0
        m = [] # cumsum
        for w in weights:
            n += w
            m.append(n)
        return [a[bisect.bisect(m, n * random.random())] for _ in range(k)]
    else:
        return [random.choice(a) for _ in range(k)]

# print(choices(['a', 'b'], weights=[0.75, 0.25], k=10))

def tiebreak(counter, default=None, key=lambda k: random.random()):
    """ Returns the dict key with max value, by default with random tie-breaking.
    """
    return max(((v, key(k), k) for k, v in counter.items()), default=(default,))[-1]

# print(tiebreak({'aa': 2, 'a': 2, 'b': 1}, key=len))

class Random(object):

    def __init__(self, seed=0, a=22695477, c=1, m=2**32):
        """ Pseudo-random number generator (LCG).
        """
        self.seed = int(seed)
        self._a = a
        self._c = c
        self._m = m
        self()

    def __call__(self):
        """ Returns a number between 0.0 and 1.0.
        """
        self.seed *= self._a
        self.seed += self._c
        self.seed %= self._m
        return float(self.seed) / self._m

# random = Random(0)
# 
# for i in range(10):
#     print(random())

#---- FILE ----------------------------------------------------------------------------------------
# Temporary files are useful when a function takes a filename, but we have the file's data instead.

class tmp(object):

    def __init__(self, s, mode='w', type=''):
        """ Returns a named temporary file containing the given string.
        """
        self._f = tempfile.NamedTemporaryFile(mode, suffix=type, delete=False)
        self._f.write(s)
        self._f.close()

    @property
    def name(self):
        return self._f.name

    def read(self):
        return self._f.read()

    def write(self, *args):
        return self._f.write(*args)

    def close(self):
        return self._f.close()

    def __enter__(self):
        return self._f

    def __exit__(self, *args):
        try: 
            os.unlink(self._f.name)
        except:
            pass

    def __del__(self):
        try: 
            os.unlink(self._f.name)
        except:
            pass

# data = '"username", "tweet", "likes"\n'
# 
# with tmp(data) as f:
#     for row in csv(f.name):
#         print(row)

def unzip(f):
    """ Returns the opened .zip file.
    """
    f = zipfile.ZipFile(f)
    f = f.open(f.namelist()[0])
    return f

##### DB ##########################################################################################

#---- CSV -----------------------------------------------------------------------------------------
# A comma-separated values file (CSV) stores table data as plain text.
# Each line in the file is a row in a table.
# Each row consists of column fields, separated by a comma.

class table(list):

    def __getitem__(self, i):
        """ A 2D list with advanced slicing: table[row1:row2, col1:col2].
        """
        if isinstance(i, tuple):
            i, j = i
            if isinstance(i, slice):
                return [v[j] for v in list.__getitem__(self, i)]
            return list.__getitem__(self, i)[j]
        return list.__getitem__(self, i)

    @property
    def html(self):
        a = ['<table>']
        for r in self:
            a.append('<tr>')
            a.extend('<td>%s</td>' % v for v in r)
            a.append('</tr>')
        a.append('</table>')
        return u'\n'.join(a)

# t = table()
# t.append([1, 2, 3])
# t.append([4, 5, 6])
# t.append([7, 8, 9])
# 
# print(t)        # [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# print(t[0])     # [1, 2, 3]
# print(t[0,0])   #  1
# print(t[:,0])   # [1, 4, 7]
# print(t[:2,:2]) # [[1, 2], [4, 5]]

csvlib.field_size_limit(1000000000)

class CSV(table):

    @classmethod
    def rows(cls, path, separator=',', encoding='utf-8'):
        """ Returns the given .csv file as an iterator of rows, 
            where each row is a list of values.
        """
        f = open(path, 'rb')
        f = (s.replace(b'\r\n', b'\n') for s in f)
        f = (s.replace(b'\r'  , b'\n') for s in f)
        f = (s.replace(b'\0'  , b''  ) for s in f) # null byte
        s = next(f, b'')
        s = s.lstrip(b'\xef\xbb\xbf') # BOM
        s = s.lstrip(b'\xff\xfe')
        f = itertools.chain((s,), f)
        e = lambda s: u(s, encoding)

        if PY3:
            f = map(e, f)
            for r in csvlib.reader(f, delimiter=separator):
                yield r
        else:
            for r in csvlib.reader(f, delimiter=separator):
                yield map(e, r)

    def __init__(self, name='', separator=',', rows=[]):
        """ Returns the given .csv file as a list of rows, 
            where each row is a list of values.
        """
        try:
            self.name      = name
            self.separator = separator
            self.extend(CSV.rows(name, separator))
        except IOError:
            pass # doesn't exist (yet)
        if rows:
            self.extend(rows)

    def save(self, name=''):
        a = []
        for r in self:
            r = ('"%s"' % u(s).replace('"', '""') for s in r)
            r = self.separator.join(r)
            a.append(r)
        f = io.open(name or self.name, 'w', encoding='utf-8')
        f.write(u'\n'.join(a))
        f.close()

    def clear(self):
        list.__init__(self, [])

csv = CSV

# data = csv('test.csv')
# data.append([1, 'hello'])
# data.save()
# 
# print(data[0,0]) # 1st cell
# print(data[:,0]) # 1st column

def col(i, a):
    """ Returns the i-th column in the given list of lists.
    """
    for r in a:
        yield r[i]

def cd(*path):
    """ Returns the path to the script calling cd() + given relative path.
    """
    f = inspect.currentframe()
    f = inspect.getouterframes(f)[1][1] 
    f = os.getcwd() if f == '<stdin>' else f
    p = os.path.realpath(f)
    p = os.path.dirname(p)
    p = os.path.join(p, *path)
    return p

path = [
    cd('kb' ), # knowledge base
    cd('lm' ), # language model
    cd('etc'),
    cd()
]

# print(cd('kb', 'en-loc.csv'))

def ls(path='*', root=path, **kwargs): 
    """ Returns an iterator of matching files & folders at the given root.
    """
    if isinstance(root, basestring):
        root = [root]
    for p in root:
        p = os.path.isabs(p) and p or cd(p)
        p = os.path.join(p, path)
        p = os.path.abspath(p)
        p = glob.glob(p, **kwargs)
        p = sorted(p)
        for f in p:
            yield f

# print(list(ls('*-pos.json.zip'))) # see tag()
# print(list(ls('*', '.')))
# print(list(ls('*.*', '/')))

# for r in csv(next(ls('en-loc.csv'))):
#     print(r[1])

#---- SQL -----------------------------------------------------------------------------------------
# A database is a collection of tables, with rows and columns of structured data.
# Rows can be edited or selected with SQL statements (Structured Query Language).
# Rows can be indexed for faster retrieval or related to other tables.

# SQLite is a lightweight engine for a portable database stored as a single file.

# https://www.sqlite.org/datatype3.html
affinity = collections.defaultdict(
    lambda : 'text'    , {
       str : 'text'    ,
   unicode : 'text'    ,
     bytes : 'blob'    ,
      bool : 'integer' ,
       int : 'integer' ,
     float : 'real'
})

def schema(table, *fields, **type):
    """ Returns an SQL CREATE TABLE statement, 
        with indices on '#'-prefixed fields.
        A field 'id' is automatically added.
    """
    s = 'create table if not exists `%s` (' % table + 'id integer primary key);'
    i = 'create index if not exists `%s_%s` on `%s` (`%s`);'
    for k in fields:
        k = re.sub(r'^\#', '', k)  # '#name' => 'name'
        v = affinity[type.get(k)]  #    str  => 'text'
        s = s[:-2] + ', `%s` %s);' % (k, v)
    for k in fields:
        if k.startswith('#'):
            s += '\n'
            s += i % ((table, k[1:]) * 2)
    return s

# print(schema('persons', '#name', 'age', age=int))

class DatabaseError(Exception):
    pass

class Database(object):

    def __init__(self, name, schema=None, timeout=10, factory=sqlite.Row):
        """ SQLite database interface.
        """
        self.connection = sqlite.connect(name, timeout)
        self.connection.row_factory = factory
        self.connection.create_function('regexp', -1, \
            lambda *v: \
                re.search(*v) is not None
        )
        if schema:
            for q in schema.strip(';').split(';'):
                q = q.strip()
                q = q + ';'
                self(q)
            self.commit()

    def __call__(self, sql, values=(), commit=False):
        """ Executes the given SQL statement.
        """
        try:
            r = self.connection.cursor().execute(sql, values)
            if commit:
                self.connection.commit()
        except Exception as e:
            raise DatabaseError('%s' % e)
        else:
            return r

    def execute(self, *args, **kwargs):
        return self(*args, **kwargs)

    def commit(self):
        return self.connection.commit()

    def rollback(self):
        return self.connection.rollback()

    def save(self):
        return self('vacuum') # reduce file size

    @property
    def id(self):
        return self('select last_insert_rowid()').fetchone()[0]

    def find1(self, *args, **kwargs):
        return self.find(*args, **kwargs).fetchone()

    def find(self, table, *fields, **filters):
        return self(*SQL_SELECT(table, *fields, **filters))

    def append(self, table,     **fields):
        b = fields.pop('commit', True)
        return self(*SQL_INSERT(table,     **fields), commit=b).lastrowid # id

    def update(self, table, id, **fields):
        b = fields.pop('commit', True)
        return self(*SQL_UPDATE(table, id, **fields), commit=b).rowcount  # int

    def remove(self, table, id, **fields):
        b = fields.pop('commit', True)
        return self(*SQL_DELETE(table, id          ), commit=b).rowcount  # int

    def __del__(self):
        try: 
            self.connection.commit()
            self.connection.close()
            self.connection = None
        except:
            pass

# db = Database(cd('test.db'), schema('persons', '#name', 'age', age=int))
# db.append('persons', name='Tom', age=30)
# db.append('persons', name='Guy', age=30)
# 
# for id, name, age in db.find('persons', age='>20'):
#     print(name, age)

def concat(a, format='%s', separator=', '):
  # concat([1, 2, 3]) => '1, 2, 3'
    return separator.join(format % v for v in a)

def op(v):
  # op([1, 2, 3]) => 'in (?, ?, ?)', (1, 2, 3)
    def esc(v):
        v = v.replace('_', '\_')
        v = v.replace('%', '\%')
        v = v.replace('*',  '%')
        return v
    if v is None:
        return 'is null', ()
    if isinstance(v, (int, float)):                 #  1
        return '= ?', (v,)
    if isinstance(v, (set, list)):                  # [1, 2, 3]
        return 'in (%s)' % concat('?' * len(v)), v
    if isinstance(v, (tuple,)):                     # (1, 2)
        return 'between ? and ?', v[:2]
    if isinstance(v, REGEX):                        # re.compile('*ly')
        return 'regexp ?', (v.pattern,)
    if isinstance(v, Op):                           # ne(1)
        return v()
    if v[:2] in ('<=', '>=', '<>', '!='):           # '<>1'
        return '%s ?' % v[:2], (v[2:],)
    if v[:1] in ('<' , '>', '='):                   # '<1'
        return '%s ?' % v[:1], (v[1:],)
    if '*' in v:                                    # '*ly'
        return "like ? escape '\\'", (esc(v),)
    else:
        return '= ?', (v,)

class Op(object):
    def __init__(self, v):
        self.v = v
    def __call__(self):
        return '= ?' , (self.v,)
class gt(Op):
    def __call__(self):
        return '> ?' , (self.v,)
class lt(Op):
    def __call__(self):
        return '< ?' , (self.v,)
class ge(Op):
    def __call__(self):
        return '>= ?', (self.v,)
class le(Op):
    def __call__(self):
        return '<= ?', (self.v,)

class ne(Op):
    def __call__(self):
        s, v = op(self.v)
        if s[:1] in '=<>':                          # ne('<=1')
            return '!' + s, v
        if s[:1] == '!':                            # ne('!=1')
            return 'like ?', '%'
        if s[:2] == 'is':                           # ne(None)
            return 'is not' + s[2:], v
        else:                                       # ne((1, 2))
            return 'not ' + s, v

def asc(k):
    return k, 'asc'
def desc(k):
    return k, 'desc'

def SQL_SELECT(table, *fields, **where):
    """ Returns an SQL SELECT statement + parameters.
    """
    s = 'select %s '        % (concat(fields, '`%s`') or '*')
    s+= 'from `%s` '        % table
    s+= 'where %s '
    s+= 'order by `%s` %s ' % where.pop('sort', ('id', 'asc'))
    s+= 'limit %s, %s;'     % where.pop('slice', (0, -1))
    k = where.keys()        # ['name', 'age']
    v = where.values()      # ['Tom*', '>10']
    v = map(op, v)          # [('like', 'Tom%'), ('>', '10')]
    v = zip(*v)             #  ('like', '>'), ('Tom%', '10')
    v = iter(v)
    x = next(v, ())
    v = next(v, ())
    v = itertools.chain(*v)
    k = iter(k.split('#')[0] for k in k)
    s = s % (concat(zip(k, x), '`%s` %s', ' and ') or 1)
    return s, tuple(v)

# print(SQL_SELECT('persons', '*', age='>10', sort='age', slice=(0, 10)))

def SQL_INSERT(table, **fields):
    """ Returns an SQL INSERT statement + parameters.
    """
    s = 'insert into `%s` (%s) values (%s);'
    k = fields.keys()
    v = fields.values()
    s = s % (table, concat(k, '`%s`'), concat('?' * len(v)))
    return s, tuple(v)

# print(SQL_INSERT('persons', name='Tom', age=10))

def SQL_UPDATE(table, id, **fields):
    """ Returns an SQL UPDATE statement + parameters.
    """
    s = 'update `%s` set %s where id=?;'
    k = fields.keys()
    v = fields.values()
    s = s % (table, concat(k, '`%s`=?'))
    return s, tuple(v) + (id,)

# print(SQL_UPDATE('persons', 1, name='Tom', age=20))

def SQL_DELETE(table, id):
    """ Returns an SQL DELETE statement + parameters.
    """
    s = 'delete from `%s` where id=?;' % table
    return s, (id,)

# print(SQL_DELETE('persons', 1))

#---- SQL ASYNC -----------------------------------------------------------------------------------
# The SQLite engine locks when writing and other read/write operations have to wait.
# A batch schedules changes in a thread-safe queue instead of writing them directly.

# A @scheduled Batch.commit() can then write the changes async,
# so there is only ever 1 thread that locks the database (~ms).

batches = {}

class Batch(Database):

    def __init__(self, name):
        self._name = name
        self.queue = batches.setdefault(name, queue.Queue())

    def append(self, *args, **kwargs):
        self.queue.put(SQL_INSERT(*args, **kwargs))

    def update(self, *args, **kwargs):
        self.queue.put(SQL_UPDATE(*args, **kwargs))

    def remove(self, *args, **kwargs):
        self.queue.put(SQL_DELETE(*args, **kwargs))

    def commit(self):
        """ Commits all pending transactions.
        """
        if not self.queue.empty():
            db = Database(self._name)
            while not self.queue.empty():
                sql = self.queue.get()
                db.execute(*sql, commit=False)
            db.commit()
            del db

    def __len__(self):
        return self.queue.qsize()

# db = Batch(cd('test.db'))
# db.append('persons', name='Tom', age=30)
# db.append('persons', name='Guy', age=30)
# 
# @scheduled(30)
# def commit():
#     Batch(cd('test.db')).commit()

#---- ENCRYPTION ----------------------------------------------------------------------------------
# The pw() function is secure enough for storing passwords; encrypt() and decrypt() are not secure.

alphanumeric = 'abcdefghijklmnopqrstuvwxyz' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + '0123456789'

def key(n=32, chars=alphanumeric):
    """ Returns a new key of length n.
    """
    return ''.join(choices(chars, k=n))

def stretch(k, n):
    """ Returns a new key of length n.
    """
    while len(k) < n:
        k += hashlib.md5(b(k)[-1024:]).hexdigest()
    return u(k[:n])

def encrypt(s, k=''):
    """ Returns the encrypted string.
    """
    k = stretch(k, len(s))
    k = bytearray(b(k))
    s = bytearray(b(s))
    s = bytearray(((i + j) % 256) for i, j in zip(s, itertools.cycle(k))) # Vigenère cipher
    s = binascii.hexlify(s)
    return u(s)

def decrypt(s, k=''):
    """ Returns the decrypted string.
    """
    k = stretch(k, len(s))
    k = bytearray(b(k))
    s = bytearray(binascii.unhexlify(s))
    s = bytearray(((i - j) % 256) for i, j in zip(s, itertools.cycle(k)))
    s = bytes(s)
    return u(s)

# print(decrypt(encrypt('hello world', '1234'), '1234'))

def pw(s, f='sha256', n=100000):
    """ Returns the encrypted string, using PBKDF2.
    """
    k = base64.b64encode(os.urandom(32)) # salt
    s = hashlib.pbkdf2_hmac(f, b(s)[:1024], k, n)
    s = binascii.hexlify(s)
    s = 'pbkdf2:%s:%s:%s:%s' % (f, n, u(k), u(s))
    return s

def pw_ok(s1, s2):
    """ Returns True if pw(s1) == s2.
    """
    _, f, n, k, s = s2.split(':')
    s1 = hashlib.pbkdf2_hmac(f, b(s1)[:1024], b(k), int(n))
    s1 = binascii.hexlify(s1)
    eq = True
    for ch1, ch2 in zip(s1, b(s)):
        eq = eq and ch1 == ch2 # contstant-time comparison
    return eq

# print(pw_ok('1234', pw('1234')))

##### ML ##########################################################################################

#---- STATISTICS ----------------------------------------------------------------------------------

def mround(v, m=1):
    """ Returns the nearest multiple m of v as a float.
    """
    m = float(m)
    return round(v / m) * m

def avg(a):
    """ Returns the average (mean) of the given values.
    """
    a = list(a)
    n = len(a) or 1
    return sum(a) / float(n)

def sd(a):
    """ Returns the standard deviation of given values.
    """
    a = list(a)
    n = len(a) or 1
    m = avg(a)
    return math.sqrt(sum((v - m) ** 2 for v in a) / n)

def peaks(a, z=1):
    """ Returns a list of indices of values that are
        more than z standard deviations above the mean.
    """
    a = list(a)
    m = avg(a)
    s = sd(a) or 1
    p = ((v - m) / s for v in a)
    p = (i for i, v in enumerate(p) if v > z)
    p = sorted(p, key=lambda i: -a[i])
    return p

# print(peaks([0, 0, 0, 10, 100, 1, 0], z=1))

#---- VECTOR --------------------------------------------------------------------------------------
# A vector is a {feature: weight} dict, with n features, or n dimensions.

# Given two points {x: 1, y: 2} and {x: 3, y: 4} in 2D,
# their distance is: sqrt((3 - 1) ** 2 + (4 - 2) ** 2).
# Distance can be calculated for points in 3D or in nD.

# Another distance metric is the angle between vectors (cosine).
# Another distance metric is the difference between vectors.
# For vectorized text cos() works well but diff() is faster.

# Vector weights are assumed to be non-negative, especially 
# when using cos(), diff(), knn(), tf(), tfidf() and freq().

def index(data=[]):
    """ Returns a dict of (id(vector), label)-items
        for the given list of (vector, label)-tuples.
    """
    return {id(v): label for v, label in data}

def distance(v1, v2):
    """ Returns the distance of the given vectors.
    """
    return sum((v1.get(f, 0) - v2.get(f, 0)) ** 2 for f in features((v1, v2))) ** 0.5

def dot(v1, v2):
    """ Returns the dot product of the given vectors.
    """
    return sum(v1.get(f, 0) * w for f, w in v2.items())

def norm(v):
    """ Returns the norm of the given vector.
    """
    return sum(w ** 2 for f, w in v.items()) ** 0.5

def cos(v1, v2):
    """ Returns the angle of the given vectors (0.0-1.0).
    """
    return 1 - dot(v1, v2) / (norm(v1) * norm(v2) or 1) # cosine distance

def diff(v1, v2):
    """ Returns the difference of the given vectors.
    """
    v1 = set(filter(v1.get, v1)) # non-zero
    v2 = set(filter(v2.get, v2))
    return 1 - len(v1 & v2) / float(len(v1 | v2) or 1) # Jaccard distance

def knn(v, vectors=[], k=3, distance=cos):
    """ Returns the k nearest neighbors from the given list of vectors.
    """
    nn = ((distance(v, x), random.random(), x) for x in vectors)
    nn = heapq.nsmallest(k, nn)
  # nn = sorted(nn)[:k]
    nn = [(1 - d, x) for d, r, x in nn]
    return nn

def reduce(v, features=set()):
    """ Returns a vector without the given features.
    """
    return {f: w for f, w in v.items() if f not in features}

def sparse(v, cutoff=0.0):
    """ Returns a vector with non-zero weight features.
    """
    return {f: w for f, w in v.items() if w > cutoff}

def binary(v, cutoff=0.0):
    """ Returns a vector with binary weights (0 or 1).
    """
    return {f: int(w > cutoff) for f, w in v.items()}

def onehot(v): # {'age': '25+'} => {('age', '25+'): 1}
    """ Returns a vector with non-categorical features.
    """
    return dict((f, w) if isinstance(w, (int, float)) else ((f, w), 1) for f, w in v.items())

def scale(v, x=0.0, y=1.0):
    """ Returns a vector with normalized weights (between x and y).
    """
    if not isinstance(v, dict):
        v = list(v)
        w = list(v.values() for v in v if v)
        w = list(itertools.chain(*w)) or [0]
        a = min(w)
        b = max(w)
        return [{f: float(v[f] - a) / (b - a or 1) * (y - x) + x for f in v} for v in v]
    else:
        return scale((v,), x, y)[0]

def unit(v):
    """ Returns a vector with normalized weights (length 1).
    """
    n = norm(v) or 1.0
    return {f: w / n for f, w in v.items()}

def normalize(v):
    """ Returns a vector with normalized weights (sum to 1).
    """
    return tf(v)

def tf(v):
    """ Returns a vector with normalized weights
        (term frequency).
    """
    n = sum(v.values())
    n = float(n or 1)
    return {f: w / n for f, w in v.items()}

def tfidf(vectors=[]):
    """ Returns an list of vectors with normalized weights
        (term frequency–inverse document frequency).
    """
    df = collections.Counter() # stopwords have higher df (a, at, ...)
    if not isinstance(vectors, list):
        vectors = list(vectors)
    for v in vectors:
        df.update(v)
    return [{f: v[f] / float(df[f] or 1) for f in v} for v in vectors]

def features(vectors=[]):
    """ Returns the set of features for all vectors.
    """
    return set().union(*vectors)

def centroid(vectors=[]):
    """ Returns the mean vector for all vectors.
    """
    v = list(vectors)
    n = float(len(v))
    return {f: sum(v.get(f, 0) for v in v) / n for f in features(v)}

def freq(a):
    """ Returns the relative frequency distribution of items in the list.
    """
    f = collections.Counter(a)
    f = collections.Counter(normalize(f))
    return f

def majority(a, default=None):
    """ Returns the most frequent item in the given list (majority vote).
    """
    f = collections.Counter(a)
    try:
        m = max(f.values())
        return random.choice([k for k, v in f.items() if v == m])
    except:
        return default

# print(majority(['cat', 'cat', 'dog']))

#---- VECTOR CLUSTERING ---------------------------------------------------------------------------
# The k-means clustering algorithm is an unsupervised machine learning method
# that partitions a given set of vectors into k clusters, so that each vector
# belongs to the cluster with the nearest center (mean).

euclidean = distance
spherical = cos

def ss(vectors=[], distance=euclidean):
    """ Returns the sum of squared distances to the center (variance).
    """
    v = list(vectors)
    c = centroid(v)
    return sum(distance(v, c) ** 2 for v in v)

def kmeans(vectors=[], k=3, distance=euclidean, iterations=100, n=10):
    """ Returns a list of k lists of vectors, clustered by distance.
    """
    vectors = list(vectors)
    optimum = None

    for _ in range(max(n, 1)):

        # Random initialization:
        g = list(shuffled(vectors))
        g = list(g[i::k] for i in range(k))[:len(g)]

        # Lloyd's algorithm:
        for _ in range(iterations):
            m = [centroid(v) for v in g]
            e = []
            for m1, g1 in zip(m, g):
                for v in g1:
                    d1 = distance(v, m1)
                    d2, g2 = min((distance(v, m2), g2) for m2, g2 in zip(m, g))
                    if d2 < d1:
                        e.append((g1, g2, v)) # move to nearer centroid
            for g1, g2, v in e:
                g1.remove(v)
                g2.append(v)
            if not e: # converged?
                break

        # Optimal solution = lowest within-cluster sum of squares:
        optimum = min(optimum or g, g, key=lambda g: sum(ss(g, distance) for g in g))
    return optimum

# data = [
#     {'woof': 1},
#     {'woof': 1},
#     {'meow': 1}
# ]
# 
# for cluster in kmeans(data, k=2):
#     print(cluster) # cats vs dogs

#---- VECTOR DIMENSIONALITY -----------------------------------------------------------------------
# Dimensionality reduction of vectors (i.e., fewer features) can speed up distance calculations.
# This can be achieved by grouping related features (e.g., covariance) or by selecting features.

def rp(vectors=[], n=100, distribution=(-1, +1)):
    """ Returns a list of vectors, each with n features.
        (Random Projection)
    """
    # Given a m x d matrix (m vectors, d features),
    # build a d x n matrix from a Gaussian distribution.
    # The dot product is the reduced vector space m x n.
    h = features(vectors)
    h = enumerate(h)
    h = map(reversed, h)
    h = dict(h) # {feature: int} hash
    r = [[random.choice(distribution) for i in range(n)] for f in h]
    p = []
    for v in vectors:
        p.append([])
        for i in range(n):
            x = 0
            for f, w in v.items():
                x += w * r[h[f]][i] # dot product
            p[-1].append(x)
    p = map(enumerate, p)
    p = map(dict, p)
    p = list(p)
    return p

def matrix(vectors=[]):
    """ Returns a 2D numpy.ndarray of the given vectors, 
        with columns ordered by sorted(features(vectors).
    """
    import numpy

    f = features(vectors)
    f = sorted(f)
    f = enumerate(f)
    f = {v: i for i, v in f}
    m = numpy.zeros((len(vectors), len(f)))
    for v, a in zip(vectors, m):
        a.put(map(f.__getitem__, v), v.values())
    return m

class svd(list):
    """ Returns a list of vectors, each with n concepts,
        where each concept is a combination of features.
        (Singular Value Decomposition)
    """
    def __init__(self, vectors=[], n=2):
        import numpy

        f  = dict(enumerate(sorted(features(vectors))))
        m  = matrix(vectors)
        m -= numpy.mean(m, 0)
        # u:  vectors x concepts
        # v: concepts x features
        u, s, v = numpy.linalg.svd(m, full_matrices=False)

        self.extend(
            sparse({   i  : w * abs(w) for i, w in enumerate(a) }) for a in u[:,:n]
        )
        self.concepts = tuple(
            sparse({ f[i] : w * abs(w) for i, w in enumerate(a) }) for a in v[:n]
        )
        self.features = normalize(
            sparse({ f[i] : 1 * abs(w) for i, w in enumerate(numpy.dot(s[:n], v[:n]))
        }))

    @property
    def cs(self):
        return self.concepts

    @property
    def pc(self):
        return self.features

pca = svd # (Principal Component Analysis)

# data = [
#     {'x': 0.0, 'y': 1.1, 'z': 1.0},
#     {'x': 0.0, 'y': 1.0, 'z': 0.0}
# ]
# 
# print(svd(data, n=2))
# print(svd(data, n=2).cs[0])
# print(svd(data, n=1).pc)

#---- FEATURES ------------------------------------------------------------------------------------
# Character 3-grams are sequences of 3 successive characters: 'hello' => 'hel', 'ell', 'llo'.
# Character 3-grams are useful, all-round features in vectorized text data,
# capturing "small words" such as pronouns, emoticons, word suffixes (-ing)
# and language-specific letter combinations (oeu, sch, tch, ...)

def chngrams(s, n=3):
    """ Returns an iterator of character n-grams.
    """
    if inspect.isgenerator(s):
        s = list(s)
    for i in range(len(s) - n + 1):
        yield s[i:i+n] # 'hello' => 'hel', 'ell', 'llo'

def ngrams(s, n=2):
    """ Returns an iterator of word n-grams.
    """
    if isinstance(s, basestring):
        s = s.split()
    for w in chngrams(s, n):
        yield tuple(w)

def skipgrams(s, n=5):
    """ Returns an iterator of (word, context)-tuples.
    """
    if isinstance(s, basestring):
        s = s.split()
    for i, w in enumerate(s):
        yield w, tuple(s[max(0,i-n):i] + s[i+1:i+1+n])

def vectorize(s, features=('ch3',)): # (vector)
    """ Returns a dict of character trigrams in the given string.
        Can be used for Perceptron.train(v(s)) or .predict(v(s)).
    """
  # s = tokenize(s)
  # s = s.lower()
    v = collections.Counter()
    v[''] = 1 # bias
    for f in features:
        f = f.lower()
        n = f[-1]
        n = int(n)
        if f[0] == '^': # '^1'
            v[f + s[:+n]] = 1
        if f[0] == '$': # '$1'
            v[f + s[-n:]] = 1
        if f[0] == 'c': # 'c1'
            v.update(chngrams(s, n))
        if f[0] == 'w': # 'w1'
            v.update(' '.join(w) for w in ngrams(s, n))
        if f[0] == '%':
            v.update(style(s))
  # v = normalize(v)
    return v

v = vec = vectorize

def style(s, bins=[0.01 * i for i in range(0, 10)] +
                  [0.10 * i for i in range(1, 10)]):
    """ Returns a dict of stylistic features in the given string 
        (word length, sentence length, capitalization, shouting).
    """
    v = {}
    for f, w in (
      ('+', u'😄|😅|😂|👍|❤️'),
      ('-', u'🙄|😢|😭|😡|💔'),
      ('~', r'[\w](?=\w{5})'  ),  # word length
      ('_', r'[\s]+'          ),  # words
      ('.', r'[.!?]+'         ),  # sentences
      ('^', r'[A-Z]'          ),  # uppercase
      (':', r'[-=:;,/%"()]'   ),  # punctuation
      ('!', r'[!]{2}'         )): # shouting
        a = re.findall(w, s)
        i = len(a)
        j = len(s) or 1
        j = float(j)
        w = i / j
        for b in bins:
            if w >= b:
                v['%s %.2f' % (f, b)] = 1
    return v

# data = []
# for id, tweet, date in csv(cd('spam.csv')):
#     data.append((v(tweet), 'spam'))
# for id, tweet, date in csv(cd('real.csv')):
#     data.append((v(tweet), 'real'))
# 
# p = Perceptron(examples=data, n=10)
# p.save(open('spam-model.json', 'w'))
# 
# print(p.predict(v('Be Lazy and Earn $3000 per Week'))) # {'real': 0.15, 'spam': 0.85}

#---- FEATURE SELECTION ---------------------------------------------------------------------------
# Feature selection identifies the best features, by evaluating their statistical significance.

def fc(data=[]): # (feature / label count)
    """ Returns a {feature: {label: count}} dict 
        for the given list of (vector, label)-tuples.
    """
    p = collections.defaultdict(
        collections.Counter)
    for v, label in data:
        for f in v:
            p[f][label] += 1
    return p

def pp(data=[]): # (posterior probability)
    """ Returns a {feature: {label: frequency}} dict 
        for the given list of (vector, label)-tuples.
    """
    f1 = collections.defaultdict(float) # {label: count}
    f2 = collections.defaultdict(float) # {feature: count}
    f3 = collections.defaultdict(float) # {feature, label: count}
    p  = collections.Counter()
    for v, label in data:
        f1[label] += 1
    for v, label in data:
        for f in v:
            f2[f] += 1
            f3[f, label] += 1 / f1[label]
    for label in f1:
        for f in f2:
            p.setdefault(f, {})[label] = f1[label] / f2[f] * f3[f, label]
    return p

def fsel(data=[]): # (feature selection, using chi2)
    """ Returns a {feature: p-value} dict 
        for the given list of (vector, label)-tuples.
    """
    from scipy.stats import chi2_contingency as chi2

    f1 = collections.defaultdict(float) # {label: count}
    f2 = collections.defaultdict(float) # {feature: count}
    f3 = collections.defaultdict(float) # {feature, label: count}
    p  = collections.Counter()
    for v, label in data:
        f1[label] += 1
    for v, label in data:
        for f in v:
            f2[f] += 1
            f3[f, label] += 1
    for f in f2:
        p[f] = chi2([[f1[label] - f3[f, label] or 0.1 for label in f1],
                     [            f3[f, label] or 0.1 for label in f1]])[1]
    return p

# data = [
#     ({'yawn': 1, 'meow': 1}, 'cat'),
#     ({'yawn': 1           }, 'dog')] * 10
# 
# bias = pp(data)
# 
# for f, p in fsel(data).items():
#     if p < 0.01:
#         print(f)
#         print(top(bias[f]))
#         # 'meow' is significant (always predicts 'cat')
#         # 'yawn' is independent (50/50 'dog' and 'cat')

#---- MODEL ---------------------------------------------------------------------------------------
# The Model base class is inherited by Perceptron, DecisionTree, ...

# Different algorithms produce models with different properties.
# Some are simple and fast, such as the probalistic Naive Bayes.
# Some are complex and slow, such as Decision Trees.

# Simple approaches have the risk of underfitting (= high bias).
# Complex approaches have the risk of overfitting (= high variance).

# Underfitting means that no patterns in the data are discovered.
# Overfitting means that useless patterns (noise) are discovered.

# Models that under/overfit do not generalize well to other data.

# Some algorithms try to prevent overfitting (and lower variance)
# for example by averaging weights (Perceptron) or with ensembles
# of multiple submodels (Random Forest).

# Some algorithms (Perceptron & Naive Bayes) use online learning,
# meaning that they can be trained on-the-fly.

# Some algorithms (Percetron, Naive Bayes, Decision Tree) do not 
# use feature weights (they just use 1) but others do (SGD, KNN).

class Model(object):

    def __init__(self, examples=[], **kwargs):
        self.labels = {}

    def __repr__(self):
        return self.__class__.__name__ + '()'

    def train(self, v, label=None):
        raise NotImplementedError

    def predict(self, v):
        raise NotImplementedError

    # Model.test() returns (precision, recall) for given test set:

    def test(self, data=[], target=None):
        if target is None:
            p = (test(self, label, data) for label in self.labels)
            p = zip(*p)
            p = map(avg, p) # macro-average
            p = tuple(p)
            return p
        else:
            return test(self, target, data)

    # Model.save() writes a serialized JSON string to a file-like.
    # Model.load() deserializes it and returns the right subclass.
    # Models that have attributes which can't be serialized, e.g.,
    # ensembles, need to override the encoder and decoder methods.

    def save(self, f):
        m = self._encode(self) # {'class': 'Model', 'labels': {}}
        m = json.dumps(m)
        f.write(m)

    @classmethod
    def load(cls, f):
        m = f.read()
        m = json.loads(m)
        m = globals().get(m.get('class'), cls)._decode(m)
        return m

    @classmethod
    def _encode(cls, model):
        """ Returns the given Model as a dict.
        """
        return dict(vars(model), **{'class': model.__class__.__name__})

    @classmethod
    def _decode(cls, dict):
        """ Returns the given dict as a Model subclass.
        """
        m = dict.pop('class', None)
        m = globals().get(m, cls)() # 'Model' => Model
        for k, v in dict.items():
            try:
                getattr(m, k).update(v) # defaultdict?
            except:
                setattr(m, k, v)
        return m

def fit(*args, **kwargs):
    m = kwargs.pop('model', NN) # 'KNN|NN|NB|DT|RF|RB|SGD'
    m = globals().get(m, m)
    try:
        return Model.load(args[0])
    except:
        return m(*args, **kwargs)

#---- PERCEPTRON ----------------------------------------------------------------------------------
# The Perceptron or single-layer neural network is a supervised machine learning algorithm.
# Supervised machine learning uses labeled training examples to infer statistical patterns.
# Each example is a dict of feature weights, with a label, e.g., ({'lottery': 1}, 'spam').

# The Perceptron takes a list of examples and learns what features are associated with what labels.
# The resulting "model" can then be used to predict the label of new examples.

def iavg(x, m=0.0, sd=0.0, n=0):
    """ Returns the iterative (mean, standard deviation, number of samples).
    """
    n += 1
    v  = sd ** 2 + m ** 2 # variance
    v += (x ** 2 - v) / n
    m += (x ** 1 - m) / n
    sd = (v - m ** 2) ** 0.5
    return m, sd, n

# p = iavg(1)     # (1.0, 0.0, 1)
# p = iavg(2, *p) # (1.5, 0.5, 2)
# p = iavg(3, *p) # (2.0, 0.8, 3)
# p = iavg(4, *p) # (2.5, 1.1, 4)
# 
# print(p)
# print(sum([1,2,3,4]) / 4.)

def softmax(p, a=1.0):
    """ Returns a dict with float values that sum to 1.0
        (using generalized logistic regression).
    """
    if p:
        a = a or 1
        v = p.values()
        v = [x / a for x in v]
        m = max(v)
        e = [math.exp(x - m) for x in v] # prevent overflow
        s = sum(e)
        v = [x / s for x in e]
        p = dict(zip(p.keys(), v))
    return p

# print(softmax({'cat': +1, 'dog': -1})) # {'cat': 0.88, 'dog': 0.12}
# print(softmax({'cat': +2, 'dog': -2})) # {'cat': 0.98, 'dog': 0.02}

def top(p, default=None):
    """ Returns a (key, value)-tuple with the max value in the dict.
    """
    try:
        v = max(p.values())
        k = [k for k in p if p[k] == v]
        k = random.choice(k)
        return k, v
    except ValueError:
        return default

# print(top({'cat': 1, 'dog': 2})) # ('dog', 2)

class Perceptron(Model):

    def __init__(self, examples=[], n=10, **kwargs):
        """ Single-layer averaged perceptron learning algorithm.
        """
        # {label: count}
        # {label: {feature: (weight, weight sum, timestamp)}}
        self.labels  = collections.defaultdict(int)
        self.weights = collections.defaultdict(dict)

        self._t = 1
        self._p = iavg(0)

        for _ in range(n):
            for v, label in shuffled(examples):
                self.train(v, label)

    def train(self, v, label=None):

        def cumsum(label, f, i, t):
            # Accumulate average weights (prevents overfitting).
            # Keep running sum + time when sum was last updated.
            # http://www.ciml.info/dl/v0_8/ciml-v0_8-ch03.pdf
            w = self.weights[label].setdefault(f, [0, 0, 0])
            w[0] += i
            w[1] += w[0] * (t - w[2])
            w[2]  = t

        self.labels[label] += 1

        guess, p = top(self.predict(v, normalize=False))
        if guess != label:
            for f, w in v.items():
                if w > 0:
                    # Error correction:
                    cumsum(label, f, +1, self._t)
                    cumsum(guess, f, -1, self._t)
            self._t += 1

        self._p = iavg(abs(p), *self._p) # (mean, sd, n)

    def predict(self, v, normalize=True):
        """ Returns a dict of (label, probability)-items.
        """
        p = dict.fromkeys(self.labels, 0.0)
        t = float(self._t)
        for label, features in self.weights.items():
            n = 0
            for f in v:
                if f in features and v[f] > 0:
                    w = features[f]
                    n = n + (w[1] + w[0] * (t - w[2])) / t
            p[label] = n

        if normalize:
            # 1. Divide values by avg + sd (-1 => +1)
            # 2. Softmax to values between 0.1 => 0.9
            #    (with < 0.1 and > 0.9 for outliers)
            p = softmax(p, a=(self._p[0] + self._p[1]))
        return p

NN = NeuralNetwork = Perceptron

# p = Perceptron(examples=[
#     ({'woof':1, 'bark':1}, 'dog'),
#     ({'meow':1, 'purr':1}, 'cat')], n=10)
# 
# print(p.predict({'meow':1}))
# 
# p.save(open('model.json', 'w'))
# p = Perceptron.load(open('model.json'))

#---- STOCHASTIC GRADIENT DESCENT -----------------------------------------------------------------
# The Stochastic Gradient Descent model is similar to Perceptron, but uses feature weights.
# The labels must be 0 or 1.

class StochasticGradientDescent(Model):

    def __init__(self, examples=[], n=10, rate=0.01, **kwargs):
        """ Stochastic Gradient Descent learning algorithm.
        """
        self.labels  = collections.defaultdict(int)
        self.weights = collections.defaultdict(float)
        self.rate    = rate # learning rate

        for _ in range(n):
            for v, label in shuffled(examples):
                self.train(v, label)

    def train(self, v, label=None):
        assert label in (0, 1)
        self.labels[label] += 1

        w = self.weights
        p = self.predict(v)
        e = self.rate * (label - top(p)[0])
        for f in v:
            w[f] += e * v[f]
        w[''] += e # bias

    def predict(self, v):
        """ Returns a dict of (label, probability)-items.
        """
        p = self.weights[''] + sum(
            self.weights[f ] * v[f] for f in v)
        p = {0: -p, 1: +p}
        p = softmax(p)
        return p

    @classmethod
    def _decode(cls, m):
        m['labels'] = {int(k): v for k, v in m['labels'].items()}
        m = Model._decode(m)
        return m

SGD = StochasticGradientDescent

#---- BERT ----------------------------------------------------------------------------------------
# The Bert model is a pre-trained, multilingual, Large Language Model (LLM) that can be fine-tuned.
# The Bert model has a commit() method that will train queued examples on CPU or GPU (~3x speedup),
# iteratively minimizing loss. It can take an optional callback(model, loss=[]) for early stopping,
# and will be called automatically on save(). It can also be called manually, for example with n=1,
# and then saved and evaluated as a "checkpoint" before training another epoch.

# The labels must be 0 or 1, and the training examples must be str.

class Bert(Model):

    def __init__(self, examples=[], n=4, rate=0.0001, base='distilbert-base-multilingual-cased', **kwargs):
        """ Bidirectional Encoder Representations from Transformers learning algorithm
        """
        import torch # 2.1+
        import transformers; transformers.logging.set_verbosity_error() # 4.25+

        self._d = kwargs.get('device') or 'cpu' # cuda | mps
        self._t = transformers.AutoTokenizer
        self._m = transformers.AutoModelForSequenceClassification

        self._t = self._t.from_pretrained(base)
        self._m = self._m.from_pretrained(base, num_labels=2)
        self._m = self._m.to(self._d)

        self._f = torch.tensor
        self._o = torch.optim.AdamW(self._m.parameters(), lr=rate) # SGD
        self._r = torch.optim.lr_scheduler.LinearLR(self._o, 1.0, 0.1, n) # learning rate decay
        self._n = n
        self._q = list(examples) # [(str, int)]

        self.labels = {0: 1, 1: 1}

    def train(self, v, label=None):
        self._q.append((v, label)) # lazy
        self.labels[label] = 1

    def _v(self, x):
        return self._t(x, truncation=True, padding='max_length', return_tensors='pt') # 512 tokens

    def commit(self, n=4, batch=16, callback=None, debug=True):
        if self._q:
            for i in range(n): # epochs
                self._r.step(i)
              # print(self._r.get_last_lr())
                v = shuffled(self._q)
                v = zip(*v)
                x = next(v)
                y = next(v)
                x = list(x)
                y = list(y)
                x = self._v(x)
                y = self._f(y) # tensor(y)
                x = x.to(self._d)
                y = y.to(self._d)
                t = time.time()
                a = []
                x['labels'] = y
                for j in range(0, len(self._q), batch):
                    e = self._m(**{k: v[j:j+batch] for k, v in x.items()}).loss # mini-batch
                    e.backward()
                    a.append(float(e))
                    self._o.step() # backpropagation
                    self._o.zero_grad()
                if callback and callback(self, a): # early stopping
                    break
                if debug:
                    print('loss=%.3f time=%.1fs' % (avg(a), time.time() - t))
            self._q = []

    def predict(self, v):
        """ Returns a dict of (label, probability)-items.
        """
        v = self._v(v).to(
            self._d)
        p = self._m(**v).logits[0] # ~0.1s
        p = enumerate(p)
        p = dict(p)
        p = softmax(p)
        return p

    def test(self, *args, **kwargs):
        return Model.test(self, *args, **kwargs)

    def save(self, path, **kwargs):
        self.commit(self._n, **kwargs)
        self._t.save_pretrained(path)
        self._m.save_pretrained(path)

    @classmethod
    def load(cls, path):
        return cls(base=path)

# examples = [
#     ('dogs say woof', 0),
#     ('cats say meow', 1),
# ]
# m = Bert(examples, device='cpu')
# m.commit(n=1, batch=32, callback=lambda m, loss: loss[-1] < 0.1, debug=True)
# m.save('m1')
# m = Bert.load('m1')

#---- NAIVE BAYES ---------------------------------------------------------------------------------
# The Naive Bayes model is a simple alternative for Perceptron (it trains very fast).
# It is based on the likelihood that a given feature occurs with a given label.

# The probability that something big and bad is a wolf is: 
# p(big|wolf) * p(bad|wolf) * p(wolf) / (p(big) * p(bad)). 

# So it depends on the frequency of big wolves, bad wolves,
# other wolves, other big things, and other bad things.

class NaiveBayes(Model):
    
    def __init__(self, examples=[], **kwargs):
        """ Binomial Naive Bayes learning algorithm.
        """
        # {label: count}
        # {label: {feature: count}}
        self.labels  = collections.defaultdict(int)
        self.weights = collections.defaultdict(dict)

        for v, label in examples:
            self.train(v, label)

    def train(self, v, label=None):
        for f, w in v.items():
            if w > 0:
                try:
                    self.weights[label][f] += 1
                except KeyError:
                    self.weights[label][f]  = 1 + 0.1 # smoothing
        self.labels[label] += 1

    def predict(self, v):
        """ Returns a dict of (label, probability)-items.
        """
        p = dict.fromkeys(self.labels, 0.0)
        for x in self.labels:
            n =  self.labels[x]
            w = (self.weights[x].get(f, 0.1) / n for f, w in v.items() if w > 0)
            w = map(math.log, w) # prevent underflow
            w = sum(w)
            w = math.exp(w) 
            w = w * n 
            w = w / sum(self.labels.values())
            p[x] = w

        s = sum(p.values()) or 1
        for label in p:
            p[label] /= s
        return p

NB = NaiveBayes

#---- K-NEAREST NEIGHBORS --------------------------------------------------------------------------
# The k-Nearest Neighbors model stores all training examples (memory-based or "lazy").
# Predicting the label of a new example is then based on the distance to each of them.

class KNN(Model):
    
    def __init__(self, examples=[], **kwargs):
        """ k-Nearest Neighbors learning algorithm.
        """
        self.labels   = collections.defaultdict(int)
        self.examples = []

        for v, label in examples:
            self.train(v, label)

    def train(self, v, label=None):
        self.examples.append((v, label))
        self.labels[label] += 1

    def predict(self, v, k=5, distance=cos):
        """ Returns a dict of (label, probability)-items.
        """
        p = dict.fromkeys(self.labels, 0.0)
        # randomly breaks equidistant ties:
        q = ((distance(v, x), random.random(), label) for x, label in self.examples)
        q = heapq.nsmallest(k, q)
        for d, _, label in q:
            p[label] += 1

        s = sum(p.values()) or 1
        for label in p:
            p[label] /= s
        return p

# examples = [
#     ("'I know some good games we could play,' said the cat.", 'seuss' ),
#     ("'I know some new tricks,' said the cat in the hat."   , 'seuss' ),
#     ("They roared their terrible roars"                     , 'sendak'),
#     ("They gnashed their terrible teeth"                    , 'sendak'),
# ]
# v, labels = zip(*examples) # unzip
# v = map(tok, v)
# v = map(wc, v)
# v = map(unit, v)
# v = list(tfidf(v))
# m = KNN(list(zip(v, labels)))
# 
# x = wc(tok('They rolled their terrible eyes'))
# x = wc(tok("'Look at me! Look at me now!' said the cat."))
# 
# y = m.predict(x, k=1, distance=lambda v1, v2: 1 - dot(v1, v2))
# print(y)

class MLKNN(KNN):

    def predict(self, v, k=5, distance=cos):
        """ k-Nearest Neighbors learning algorithm (multi-label).
        """
        p = collections.Counter()
        n = min(k, sum(self.labels.values()))
        for labels, w in KNN.predict(self, v, k, distance).items():
            for label in labels:
                if w > 0:
                    p[label] += 1.0 / n
        return p

# m = [
#     ({'woof': 1}, ('dog', 'canine')),
#     ({'meow': 1}, ('cat', 'feline')),
#     ({'meow': 1}, ('cat',)),
# ]
# m = MLKNN(m)
# 
# print(m.predict({'meow': 1}, k=2))

#---- DECISION TREE --------------------------------------------------------------------------------
# The Decision Tree model is very easy to interpret (but it trains very slow).
# It is based on a directed graph with features as nodes and labels as leaves.

# The training data is recursively partitioned into subsets (edges)
# where the left branch has a feature and the right branch has not.
# Which feature to split next is decided by a cost function (gini).

def gini(data=[]):
    """ Returns the diversity of data labels (0.0-1.0).
    """
    p = collections.Counter(label for v, label in data)
    n = len(data)
    n = float(n or 1)
    g = 1 - sum((i / n) ** 2 for i in p.values())
    return g

# print(gini([({}, 'cat'), ({}, 'cat')])) # 0.0
# print(gini([({}, 'cat'), ({}, 'dog')])) # 0.5

class DecisionTree(Model):

    def __init__(self, examples=[], **kwargs):
        """ Decision Tree learning algorithm.
        """
        count = lambda data: collections.Counter(label for v, label in data)

        def split(data, cost=gini, min=2, max=10**10, depth=100):
            n = None
            for f in sliced(shuffled(features(v for v, label in data)), max):
                Y = []
                N = []
                for v, label in data:
                    if v.get(f, 0) > 0:
                        Y.append((v, label))
                    else:
                        N.append((v, label))
                g = cost(Y) * len(Y) / len(data) \
                  + cost(N) * len(N) / len(data)
                if not n or n[4] > g:
                  # n[0]: feature
                  # n[1]: vectors w/  feature
                  # n[2]: vectors w/o feature
                  # n[3]: sample size
                  # n[4]: cost (gini)
                    n = [f, Y, N, len(data), g]

            if n is None:
                return count(data)
            if len(data) < min:
                return count(data)
            if depth == 0 or n[4] == 0:
                n[1] = count(n[1])
                n[2] = count(n[2])
            else:
                n[1] = split(n[1], cost, min, max, depth-1)
                n[2] = split(n[2], cost, min, max, depth-1)
            return n

        self.labels = count(examples)
        self.root   = split(examples, **kwargs)

    def predict(self, v):
        """ Returns a dict of (label, probability)-items.
        """
        n = self.root
        while isinstance(n, list): # search
            if v.get(n[0], 0) > 0:
                n = n[1]
            else:
                n = n[2]

        p = dict.fromkeys(self.labels, 0.0)
        p.update(n)
        p = freq(p)
        return p

def rules(tree):
    """ Returns an iterator of edges for the given DecisionTree,
        where each edge is a (node1, node2, weight, type)-tuple.
    """
    k = {} # {id(node): int}
    q = [tree.root]
    while q:
        n1 = q.pop(0) # bfs
        for n2, e in ((n1[1], True), (n1[2], False)):
            k1 = '#%i ' % k.setdefault(id(n1), len(k))
            k2 = '#%i ' % k.setdefault(id(n2), len(k))
            if isinstance(n2, list):
                q.append(n2)
                n2 = {n2[0]: n2[3]}
            for f, w in n2.items():
                yield k1 + n1[0], k2 + f, float(w) / tree.root[3], e
                # ('#1 meow', '#2 cat', 0.5, True )
                # ('#1 meow', '#3 dog', 0.5, False)

DT = DecisionTree

#---- DECISION TREE ENSEMBLE -----------------------------------------------------------------------
# The Decision Tree Ensemble has multiple decision trees with majority vote, to correct overfitting.

class DecisionTreeEnsemble(Model):

    def __init__(self, examples=[], n=10, min=5, max=100, **kwargs):
        """ Decision Tree ensemble (Random Forest algorithm).
        """
        self.labels = collections.Counter(label for v, label in examples)
        self.trees  = []

        for _ in range(n):
            # Breiman's algorithm.
            # Random sample of features.
            # Random sample of examples (with replacement).
            t = [random.choice(examples) for e in examples]
          # t = DecisionTree(t, min=min, max=max, **kwargs)
            self.trees.append(t)

        self.trees = list(parallel(DecisionTree, self.trees, min=min, max=max, **kwargs)) # faster

    def predict(self, v):
        """ Returns a dict of (label, probability)-items.
        """
        p = collections.Counter()
        for t in self.trees:
            p += t.predict(v)
        for k, v in p.items():
            p[k] = v / len(self.trees)
        p = {k: p.get(k, 0.0) for k in self.labels}
        return p

    @classmethod
    def _encode(cls, m):
        m = Model._encode(m)
        m['trees'] = [DecisionTree._encode(t) for t in m['trees']]
        return m

    @classmethod
    def _decode(cls, m):
        m['trees'] = [DecisionTree._decode(t) for t in m['trees']]
        m = Model._decode(m)
        return m

RF = RandomForest = DecisionTreeEnsemble

# To predict from a mixed ensemble, use: centroid((m.predict(v) for m in models))

#---- RANDOM BASELINE -----------------------------------------------------------------------------
# The Random Baseline model can be used to compare a real model's performance to.

class Baseline(Model):

    def __init__(self, examples=[], **kwargs):
        """ Weighted random baseline algorithm.
        """
        self.labels = collections.Counter(label for v, label in examples)

    def train(self, v, label):
        self.labels[label] += 1

    def predict(self, v):
        p = dict.fromkeys(self.labels, 0.0)
        k = self.labels.items() # (k1, w1), (k2, w2)
        k = zip(*k)             # (k1, k2), (w1, w2)
        k = choices(*k)[0]
        p[k] = 1.0
        return p

RB = RandomBaseline = Baseline

#---- PERFORMANCE ---------------------------------------------------------------------------------
# Predicted labels will often include false positives and false negatives.
# A false positive is a real e-mail that is labeled as spam (for example).
# A false negative is a real e-mail that ends up in the junk folder.

# To evaluate how well a model deals with false positives and negatives (i.e., accuracy),
# we can use a list of labeled test examples and check the label vs. the predicted label.
# The evaluation will yield two scores between 0.0 and 1.0: precision (P) and recall (R).
# Higher precision = less false positives.
# Higher recall    = less false negatives.

# A robust evaluation of P/R is by K-fold cross-validation.
# K-fold cross-validation takes a list of labeled examples,
# and trains + tests K different models on different subsets of examples.

def balanced(data=[], n=None):
    """ Returns an iterator of (vector, label)-tuples,
        with at most n of every label.
    """
    f = collections.Counter()
    for v, label in data:
        f[label] += 1
        if n is None or f[label] <= n:
            yield v, label

def confusion_matrix(model, test=[]):
    """ Returns the matrix of labels x predicted labels, as a dict.
    """
    # { label: { predicted label: count}}
    m = collections.defaultdict(lambda: \
        collections.defaultdict(int))
    for label in model.labels:
        m[label]
    for v, label in test:
        guess, p = top(model.predict(v))
        m[label][guess] += 1
    return m

cf = confusion_matrix

def errors(model, test=[]):
    """ Returns an iterator of incorrect (vector, label, predicted).
    """
    for v, label in test:
        guess, p = top(model.predict(v))
        if guess != label:
            yield v, label, guess

def test(model, target, data=[]):
    """ Returns a (precision, recall)-tuple for the test data.
        High precision = few false positives for target label.
        High recall    = few false negatives for target label.
    """
    if isinstance(model, Model):
        m = confusion_matrix(model, data)
    if isinstance(model, dict): # confusion matrix
        m = model

    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    for x1 in m:
        for x2, n in m[x1].items():
            if target == x1 == x2:
                TP += n
            if target != x1 == x2:
                TN += n
            if target == x1 != x2:
                FN += n
            if target == x2 != x1: 
                FP += n
    return (
        TP / (TP + FP or 1),
        TP / (TP + FN or 1))

def kfoldcv(Model, data=[], k=10, weighted=False, debug=False, **kwargs):
    """ Returns the average precision & recall across labels, in k tests.
    """

    def folds(a, k=10):
      # folds([1,2,3,4,5,6], k=2) => [1,2,3], [4,5,6]
        return (a[i::k] for i in range(k))

    def wavg(a):
      # wavg([(1, 0.33), (2, 0.33), (3, 0.33)]) => 2  (weighted mean)
        return sum(v * w for v, w in a) / (sum(w for v, w in a) or 1)

    data = list(shuffled(data))
    data = list(folds(data, k))

    P = []
    R = []
    for i in range(k):
        x = data[i]
        y = data[:i] + data[i+1:]
        y = itertools.chain(*y)
        y = list(y)
        m = Model(examples=y, **kwargs)
        f = confusion_matrix(m, test=x)
        for label, n in m.labels.items():
            if not weighted:
                n = 1
            precision, recall = test(f, target=label)
            P.append((precision, n))
            R.append((recall, n))

            if debug:
                print(u'k=%i P %.1f R %.1f %s' % (i+1, precision * 100, recall * 100, label))

        if debug and set(m.labels) == {0, 1}:
            print(u'k=%i φ %+.2f\n' % (i+1, mcc(f)))

    return wavg(P), wavg(R)

def F1(P, R):
    """ Returns the harmonic mean of precision and recall.
    """
    return 2.0 * P * R / (P + R or 1)

# data = []
# for id, username, tweet, date in csv(cd('spam.csv')):
#     data.append((v(tweet), 'spam'))
# for id, username, tweet, date in csv(cd('real.csv')):
#     data.append((v(tweet), 'real'))
# 
# print(kfoldcv(Perceptron, data, k=3, n=5, debug=True)) # ~ P 0.80 R 0.80

def mcc(cf):
    """ Returns the correlation coefficient (-1.0 to +1.0)
        for a binary confusion matrix with labels 0 and 1.
    """
    TP = cf[1][1]
    TN = cf[0][0]
    FP = cf[0][1]
    FN = cf[1][0]

    return (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) or 1)
    
# print((mcc({1: {1: 3, 0: 1}, 0: {0: 4, 1: 1}}) + 1) / 2)

#---- CONFIDENCE ----------------------------------------------------------------------------------
# Predicted labels usually come with a probability or confidence score.
# However, the raw scores of Perceptron + softmax, SVM and Naive Bayes 
# do not always yield good estimates of true probabilities.

# We can use a number of training examples for calibration.
# Isotonic regression yields a function that can be used to
# map the raw scores to well-calibrated probabilities.

def pav(y=[]):
    """ Returns the isotonic regression of y
        (Pool Adjacent Violators algorithm).
    """
    y = list(y)
    n = len(y) - 1
    while 1:
        e = 0
        i = 0
        while i < n:
            j = i
            while j < n and y[j] >= y[j+1]:
                j += 1
            if y[i] != y[j]:
                r = y[i:j+1]
                y[i:j+1] = [float(sum(r)) / len(r)] * len(r)
                e = 1
            i = j + 1
        if not e: # converged?
            break
    return y

# Example from Fawcett & Niculescu (2007), PAV and the ROC convex hull:
# 
# y = sorted((
#     (0.90, 1),
#     (0.80, 1),
#     (0.70, 0),
#     (0.60, 1),
#     (0.55, 1),
#     (0.50, 1),
#     (0.45, 0),
#     (0.40, 1),
#     (0.35, 1),
#     (0.30, 0),
#     (0.27, 1),
#     (0.20, 0),
#     (0.18, 0),
#     (0.10, 1),
#     (0.02, 0)
# ))
# y = zip(*y)
# y = list(y)[1]
# print(pav(y))

class Calibrated(Model):

    def __repr__(self):
        return 'Calibrated(%s)' % repr(self._model)

    def __init__(self, model, label, data=[], **kwargs):
        """ Returns a model calibrated on the given data,
            which is a set of (vector, label)-tuples.
        """
        self._model = model
        self._label = label

        # Isotonic regression:
        y = ((model.predict(v)[label], label == x) for v, x in data)
        y = sorted(y) # monotonic
        y = zip(*y)
        y = list(y) or ((),())
        x = list(y[0])
        y = list(y[1])
        y = pav(y)
        x = [0] + x + [1]
        y = [0] + y + [1]
        f = {}
        i = 0

        # Linear interpolation:
        for p in range(100 + 1):
            p = p * 0.01
            while x[i] < p:
                i += 1
            f['%.2f' % p] = (y[i-1] * (x[i] - p) + y[i] * (p - x[i-1])) / (x[i] - x[i-1])

        self._f = f

    def predict(self, v, *args, **kwargs):
        """ Returns the label's calibrated probability (0.0-1.0).
        """
        p = self._model.predict(v, *args, **kwargs)[self._label] 
        p = self._f['%.2f' % p] + 0
        p = {self._label: p}
        return p

    @property
    def labels(self):
        return self._model.labels

    @classmethod
    def _encode(cls, m):
        m1 = Model._encode(m._model)
        m2 = {
            'class' : 'Calibrated',
            'model' : m1, 
            'label' : m._label, 
                'f' : m._f, 
        }
        return m2

    @classmethod
    def _decode(cls, m):
        m1 = Model._decode(m['model'])
        m2 = calibrate(m1, m['label'])
        m2._f = m['f']
        return m2

# Note: Calibrated won't work with kfoldcv().

calibrate = Calibrated

# data = []
# for review, polarity in csv('reviews.csv'):
#     data.append((v(review), polarity))
# 
# m = Model.load('sentiment.json')
# m = calibrate(m, '+', data)

def calibrated(examples=[], **kwargs):
    """ Returns a calibrated model.
    """
    k = kwargs.pop('label')
    m = kwargs.pop('base' , 
        kwargs.pop('model', NN))
    d = kwargs.pop('d', 0.1)
    d = max(d, 0.0)
    d = min(d, 1.0)
    e = list(shuffled(examples))
    i = len(e) * d
    i = int(i)
    a = e[i:]
    b = e[:i] # 10%
    m = fit(a, model=m, **kwargs)
    m = Calibrated(m, label=k, data=b)
    return m

# m = calibrated(data, label='+', model=NN)
# m.save('sentiment.json')

#---- AGREEMENT -----------------------------------------------------------------------------------
# Inter-rater agreement estimates the reliability of annotations, given a number of annotators.

def agreement(a=[]):
    """ Returns the agreement (0.0 to 1.0)
        in a given list of annotations,
        each a list of votes per label.
    """
    a = a or [[]]
    m = len(a)
    k = len(a[0]) # labels
    n = sum(a[0]) # voters
    p = sum(v ** 2. for example in a for v in example) # free-marginal overall (Randolph, 2008)
    p = p - (m * n)
    p = p / (m * n * (n - 1) or 1)
    return p

# print(agreement([[2, 0], [0, 2]]))

#---- EXPLAINABILITY ------------------------------------------------------------------------------
# Complex models can become "black boxes" with little insight into their decision-making process.
# Even if they are very accurate, not being able to explain themself has legal and ethical risks.

# Some things we can do:
# 1) Use sample() to surmise the training data. Does it represent the labels? Are they separable?
# 2) Use fsel() and pp() to examine significant features that occur more frequently with a label.
#    If this includes features like "and" or "@name", then the model may be overfitting on noise.
#    Decision Tree ensembles counter noise but they're slow. Trees can be examined with .graph().
# 3) Use test() with a holdout set not used for kfoldcv(). Do the performance metrics hold up?
# 4) Use test() with a cross-domain (i.e., related) dataset. Do the performance metrics hold up?
# 5) Use explain() and hilite() to review individual predictions. Do these make sense to humans?
#    This is also useful for error analysis, to uncover influential features in false positives.
# 6) Use deliberately biased input (e.g., "black woman" vs "white man") to test predicted output.

def sample(data=[]):
    """ Returns an iterator of diverse representative examples
        from the given list of (vector, label)-tuples.
    """
    # https://arxiv.org/pdf/1602.04938.pdf

    f1 = collections.defaultdict(float) # {label: count}
    f2 = collections.defaultdict(float) # {feature: count}
    f3 = collections.defaultdict(float) # {feature, label: count}
    for v, label in data:
        f1[label] += 1
    for v, label in data:
        for f in v:
            f2[f] += 1
            f3[f, label] += 1

    # Submodular pick:
    seen = set()
    while 1:
        x = 0, None
        for v, label in data:
            w = 0
            for f in v:
                if f not in seen:
                    w += f3[f, label] * len(f1)
                    w -= f2[f]
            if x[0] < w:
                x = w, v, label # greedy
        if not x[0]:
            break
        seen.update(x[1]) # nonredundant
        yield x[1:]

# data = [
#     (('A',       ) , 'X'),
#     (('A','B'    ) , 'X'),
#     (('A','B','C') , 'X'),
#     (('C','D'    ) , 'Y'),
#     (('D','E'    ) , 'Y'),
# ]
# 
# print(list(sample(data))) # (A, B), (D, E)

def cc(v, label, model):
    """ Returns a (feature, weight)-dict with relative
        feature-label correlation coefficients (-1 to +1).
    """
    p = model.predict(v)[label]
    r = dict()
    e = dict(v)

    for f, w in v.items():
        e[f] = int(w == 0)
        r[f] = p - model.predict(e)[label] # leave-one-out
        e[f] = w

    # Most biased feature has weight 1:
    n = max(map(abs, r.values())) or 1
    r = {f: w / n for f, w in r.items()}
    r = collections.Counter(r)
    return r

def explain(v, model):
    """ Returns label, probability, (feature, weight)-dict.
    """
    label, p = top(model.predict(v))
    return label, p, cc(v, label, model)

# m = Perceptron([
#     ({'woof': 1}, 'dog'),
#     ({'meow': 1}, 'cat')])
#
# print(explain(m, {'meow': 1, 'purr': 1}))

def counterfactual(v, label, model):
    """ Returns the minimal set of features to remove, 
        so that the given vector will have given label.
    """
    p = model.predict(v)[label]
    r = cc(v, label, model)
    e = set()
    for f, w in reversed(r.most_common()):
        if p < 0.5 and w < 0:
            p -= p * w
            e.add(f)
        else:
            break
    return e

# m = Perceptron([
#     ({'black': 1, '>25': 0, 'theft': 1, 'drugs': 0}, 'jail'),
#     ({'black': 1, '>25': 1, 'theft': 0, 'drugs': 1}, 'jail'),
#     ({'black': 0, '>25': 1, 'theft': 1, 'drugs': 0}, 'bail'),
#     ({'black': 0, '>25': 1, 'theft': 0, 'drugs': 1}, 'bail')])
# 
# This model is biased towards jailing black people:
# print(counterfactual({'black': 1, '>25': 1, 'drugs': 1}, 'bail', m))

def hilite(s, words={}, style='background: #ff0{:1f};', format=max):
    """ Returns a string with <mark> tags for given (word, weight)-items,
        where background opacity of marked words represents their weight.
    """
    e = '<mark style="%s">' % style, '</mark>'
    w = words
    w = w if isinstance(w, trie) else trie(w) 
    m = w.search(s, sep=None)
    s = mark(s, m, format, tag=e)
    return s

# print(hilite('loud meowing', {'loud': 1.0}))

#---- CLASSIFIER ----------------------------------------------------------------------------------
# The Classifier class combines a concrete model with its training data and vectorization function.

class Classifier(object):

    def data(self):
        raise NotImplemented # [(example, label)]

    def v(self, v):
        raise NotImplemented # {}

    def vectorized(self):
        return [(self.v(v), k) for v, k in self.data()]

    def __init__(self, path, *args, **kwargs): # model=NN
        """ Returns a classifier with a trained model.
        """
        try:
            with open(path, 'r') as f:
                self.model = Model.load(f)
        except:
            with open(path, 'w') as f:
                self.model = fit(self.vectorized(), *args, **kwargs)
                self.model.save(f)

    def __call__(self, v):
        """ Returns the predicted (label, confidence).
        """
        return top(self.predict(v))

    def predict(self, v, *args, **kwargs):
        """ Returns the predicted {label: confidence, ...}.
        """
        v = self.v(v)
        k = self.model.predict(v, *args, **kwargs)
        return k

    def test(self, k=10, **kwargs):
        """ Returns the average (precision, recall).
        """
        if isinstance(self.model, Calibrated):
            m = self.model._model
        else:
            m = self.model
        return kfoldcv(m.__class__, self.vectorized(), k=k, **kwargs)

# class Toxicity(Classifier):
#     def data(self):
#         return (('fools', 1), ('kittens', 0))
#     def v(self, s):
#         return vec(s, features=('ch3', 'w1'))
#     def __call__(self, s):
#         return int(super().__call__(s)[0]) > 0
# 
# toxic = Toxicity('toxicity.json', model=NN)
# 
# print(toxic('foolish'))
# print(toxic('kitteh'))

# class Gender(Classifier):
#     def data(self):
#         return (r[:2] for r in csv(cd('kb', 'en-per.csv')))
#     def v(self, s):
#         return vec(s, features=('c2', 'c3', '$2'))
#     def __call__(self, s):
#         return round(super().predict(s)['m'], 2)
# 
# male = Gender('gender.json', model=calibrated, base=NB, label='m')
# 
# print(male('Alice'))
# print(male('Kitty'))
# print(male('Bob'))
# print(male('Taylor'))
# print(male('Thomas'))

##### NLP #########################################################################################

#---- TEXT GENERATION -----------------------------------------------------------------------------
# The cfg() function (Context-Free Grammar) recursively replaces placeholders in a string.
# The replacements are chosen from a list of (placeholder) strings, generating variations.

def cfg(s, rules={}, placeholder=r'@([\w+]+)', format=lambda s: s, depth=10):
    """ Returns an iterator of strings with replaced placeholders.
    """
    def closure(s, n):
        def replace(m):
            try:
                s = m.group(1)        #  '@hungry'
                s = rules[s]          # ['starved', ...]
                s = random.choice(s)  #  'starved'
                s = u(s)              # u'starved'
                s = closure(s, n-1)
            except:
                s = m.group(0)
            return s
        if n <= 0:
            return s
        else:
            return re.sub(placeholder, replace, s)

    while 1:
        yield format(closure(s, depth))

# print(next(cfg('The cat is @hungry!',
#     rules={
#         'very': (
#             'extremely'    ,
#             'immensely'    ,
#         ),
#         'hungry': (
#             'ravenous'     ,
#             'starving'     ,
#             '@very hungry' ,
#         ),
#     }
# )))

#---- TEXT SEARCH ---------------------------------------------------------------------------------
# The trie() function maps a dict of strings (i.e., a lexicon) to a tree of chars, for fast search.

Match = collections.namedtuple('Match', ('start', 'end', 'word', 'tag'))

def subs(m1, m2):
    """ Returns True if the first match subsumes (=contains) the second.
    """
    return m1[0] <= m2[0] and m1[1] >= m2[1] and len(m1[2]) > len(m2[2])

# print(subs((0, 11, 'superficial', -1), (0, 5, 'super', +1)))

class trie(dict):

    def __init__(self, lexicon={}):
        """ Returns a trie for the dict of str keys.
        """
        for k, v in lexicon.items():
            n = self
            for k in k:
                if k not in n:
                    n[k] = {} # {'w': {'o': {'w': {None: +0.5}}}}
                n = n[k]
            n[None] = v

    def search(self, s, sep=lambda ch: not ch.isalpha(), etc=None):
        """ Returns an iterator of (i, j, k, v) matches 
            for keys bounded by the separator (or None).
        """
        x = (etc or '')[0:1] # *
        y = (etc or '')[1:2] # .
        z = (etc or '')[2:3] # ?
        n = ((0, self),)
        m = dict(self)
        m.pop(x, None) # w/o lookbehind 

        for j, c in enumerate(s + ' '): 
            _ = []
            a = []
            b = not sep    \
                 or sep(c) \
                 or j == len(s) # non-greedy

            for i, n in n:
                # Find matches:
                if b and i < j and None in n:
                    yield Match(i, j, s[i:j], n[None])
                if b and i < j and z in n and None in n[z]: # ?
                    yield Match(i, j, s[i:j], n[z][None])
                if b and i < j and x in n and None in n[x]: # *
                    yield Match(i, j, s[i:j], n[x][None])
                # Find branches:
                if x in n:
                    a.append((i, {x: n[x]}))  # cat = c*
                if x in n and c in n[x]:
                    a.append((i, n[x][c]))    # cat = c*t
                if x in n and c in n[x]:
                    _.append((i, n[x][c]))    # c t = c *
                if y in n:
                    a.append((i, n[y]))       # cat = c.t
                if z in n:
                    a.append((i, n[z]))       # cat = cat?
                if z in n and c in n[z]:
                    a.append((i, n[z][c]))    # cat = ca?t
                if c in n:
                    a.append((i, n[c]))       # cat = cat
                if c in n:
                    _.append((i, n[c]))       # c t = c t
            if b:
                # Find breaks:
                if sep:
                    a.clear()
                    a.extend(_) # (phrases)
                    a.append((j + 1, self))
                else:
                    a.append((j + 1, m))
            n = a

# print(list(trie({'wow': +0.5, 'wtf': -0.5}).search('oh wow!')))

# print(list(trie({'cat*': True}).search('catnip cat', etc='*')))

# print(list(trie({'cat *': True}).search('cat fancy', etc='*')))

#---- TEXT MARKUP ---------------------------------------------------------------------------------

class Formatter(string.Formatter):
    """ Returns a string formatter with custom specifiers.
    """
    def format_field(self, v, k):
        if k == '1f':
            k, v = 'x', int(v * 15) # 1.0 = 'f'
        return string.Formatter.format_field(self, v, k)

formatter = Formatter()

# print(formatter.format('color: #ff0{:1f};', 0.5))

def mark(s, matches=[], format=lambda w: ' '.join(w), tag=('<mark class="{}">', '</mark>')):
    """ Returns a HTML string with <mark> tags for matches.
    """
    m1 = collections.defaultdict(set)
    m2 = []
    # Merge class by character:
    for i, j, k, v in matches:
        if i is None:
            i = 0
        if j is None:
            j = len(s)
        for i in range(i, j):
            m1[i].add(v)
    # Merge character by class:
    for i, v in sorted(m1.items()):
        if not m2 or m2[-1][0] != v or m2[-1][1][1] < i:
            m2.append((v, [i, i]))
        m2[-1][1][1] += 1

    f = u''
    o = 0
    for v, (i, j) in m2:
        if i < len(s):
            v = sorted(v)
            v = format(v)
            f = f + encode(s[o:i])
            f = f + formatter.format(u(tag[0]), v)
            f = f + encode(s[i:j])
            f = f + u(tag[1])
            o = j
    f += encode(s[o:])
    return f

# s = 'Klaatu stirs!'
# t = trie({'Klaatu': 'cat'})
# m = t.search(s)
# 
# print(mark(s, m))

#---- TEXT ----------------------------------------------------------------------------------------

diacritics = u"àáâãäåǟāąæçćčςďḑèéêëēěęģìíîïīįłķќĺļľņńñňѝйòóőôõȭöȯȱōðøœþŕřŗśšťțùúűûüůūųẁẃẅŵỳýÿŷўźžż"

URL = re.compile(r'(https?://.*?|www\.*?\.[\w.]+|[\w-]+\.(?:com|net|org))(?=\)?[,;:!?.]*(?:\s|$))')
REF = re.compile(r'[\w.-]*@[\w.-]+', flags=re.U)

def similarity(s1, s2, f=ngrams, n=2):
    """ Returns the degree of lexical similarity (0.0-1.0)
        measured by overlapping word or character n-grams.
    """
    v1 = collections.Counter(f(s1, n))
    v2 = collections.Counter(f(s2, n))
    v1 = set((k, i) for k in v1 for i in range(v1[k]))
    v2 = set((k, i) for k in v2 for i in range(v2[k]))

    return 2 * len(v1 & v2) / float(len(v1) + len(v2) or 1) # Sørensen–Dice coefficient

sim = similarity

# s1 = 'the black cat sat on the mat'
# s2 = 'the white cat sat on the mat'
# print(sim(s1, s2, n=3))

# print(sim('wow', 'wowow', chngrams))
# print(sim('lol', 'wowow', chngrams))

def readability(s):
    """ Returns the readability of the given string (0.0-1.0).
    """
    # Flesch Reading Ease; Farr, Jenkins & Patterson's metric.

    def syllables(w, v="aeiouy"):
      # syllables('several') => 2, se-ve-ral
        if w.endswith('e'):
            w = w[:-1]
        return sum(not ch1 in v and \
                       ch2 in v for ch1, ch2 in zip(w, w[1:])) or 1

    s = s.lower()
    s = s.strip()
    s = s.strip('.!?()\'"')
    s = re.sub(r'[\-/]+', ' ', s)
    s = re.sub(r'[\s,]+', ' ', s)
    s = re.sub(r'[.!?]+', '.', s)
    s = re.sub(r'(\. )+', '. ',s)
    y = map(syllables, s.split()) # syllables
    w = max(1, len(s.split(' '))) # words
    s = max(1, len(s.split('.'))) # sentences
    r = 1.599 * sum(n == 1 for n in y) * 100 / w - 1.015 * w / s - 31.517
    r = 0.01 * r 
    r = w == 1 and 1.0 or r
    r = max(r, 0.0)
    r = min(r, 1.0)
    return r

def anon(s):
    """ Returns the string with no URLs and @usernames.
    """
    s = re.sub(URL, 'http://', s)
    s = re.sub(REF, '@[...]', s)
    return s

# print(anon('@Textgain (https://www.textgain.com)'))

def collapse(s, separator=' '):
    """ Returns the string with collapsed whitespace.
    """
    s = s.strip()
    s = s.split()
    s = separator.join(s)
    return s

# print(collapse('wow\n  nice '))

def cap(s):
    """ Returns the string with capitalized sentences.
    """
    s = re.split(r'([.!?]\s+)', s)
    s = ''.join(s[:1].upper() + s[1:] for s in s)
    return s

# print(cap('wow! nice.'))

def sep(s):
    """ Returns the string with separated punctuation marks.
    """
    s = s.replace( ',',  ' , ')
    s = s.replace( ';',  ' ; ')
    s = s.replace( ':',  ' : ')
    s = s.replace( '.',  ' . ') # fast, naive
    s = s.replace( '!',  ' ! ')
    s = s.replace( '?',  ' ? ')
    s = s.replace( '(',  ' ( ')
    s = s.replace( ')',  ' ) ')
    s = s.replace( '"',  ' " ')
    s = s.replace(u"“", u" “ ")
    s = s.replace(u"”", u" ” ")
    s = s.replace(u"‘", u" ‘ ")
    s = s.replace(u"’", u" ’ ")
    s = s.replace( "' ", " ' ")
    s = s.replace( " '", " ' ")
    s = s.replace("'s "," 's ")
    s = s.replace(';  )', ';)')
    s = s.replace(':  )', ':)')
    s = s.replace(':  (', ':(')
    s = s.replace(' : /', ':/')
    s = s.replace('www . ', 'www.')
    s = s.replace(' . com', '.com')
    s = s.replace(' . net', '.net')
    s = s.replace(' . org', '.org')
    s = s.split()
    s = ' '.join(s)
    return s

# print(sep('"Wow!" :)'))

def detag(s):
    """ Returns the string with no HTML tags.
    """
    class Parser(HTMLParser, list):
        def handle_data(self, s):
            self.append(s)
        def handle_entityref(self, s):
            self.append('&')
        def __call__(self, s):
            self.feed(u(s).replace('&', '&amp;'))
            self.close()
            return ''.join(self)

    return Parser()(s)

# print(detag('<a>a</a>&<b>b</b>'))

def destress(s, replace={}):
    """ Returns the string with no diacritics.
    """
    for k, v in replace.items():
        s = s.replace(k, v)
    for k, v in {
     u'Æ' : 'Ae',
     u'Œ' : 'Oe',
     u'æ' : 'ae',
     u'œ' : 'oe',
     u'ä' : 'ae',
     u'ö' : 'oe',
     u'ü' : 'ue',
     u'ß' : 'ss',
     u'Þ' : 'Th',
     u'þ' : 'th',
     u'Đ' : 'Dj',
     u'đ' : 'dj',
     u'Ł' : 'L' ,
     u'ł' : 'l' ,
     u'Ø' : 'O' ,
     u'ø' : 'o' ,
     u'i̇' : 'i' ,
     u'“' : '"' ,
     u'”' : '"' ,
     u'„' : '"' ,
     u'‘' : "'" ,
     u'’' : "'" ,
     u'⁄' : '/' ,
     u'¿' : '?' ,
     u'¡' : '!' ,
     u'«' : '<<',
     u'»' : '>>'}.items():
        s = s.replace(k, v)
    f = unicodedata.combining             # f('´') == 0
    s = unicodedata.normalize('NFKD', s)  # é => e + ´
    s = ''.join(ch for ch in s if not f(ch))
    return s

# print(destress(u'pâté')) # 'pate'

def deflood(s, n=3):
    """ Returns the string with no more than n repeated characters.
    """
    if n == 0:
        return s
    return re.sub(r'((.)\2{%s,})' % (n-1), lambda m: m.group(1)[0] * n, s)
    
# print(deflood('Woooooow!!!!!!', n=3)) # 'Wooow!!!'

def decamel(s, separator="_"):
    """ Returns the string with CamelCase converted to underscores.
    """
    s = re.sub(r'(.)([A-Z][a-z]{2,})', '\\1%s\\2' % separator, s)
    s = re.sub(r'([a-z0-9])([A-Z])'  , '\\1%s\\2' % separator, s)
    s = re.sub(r'([A-Za-z])([0-9])'  , '\\1%s\\2' % separator, s)
    s = re.sub(r'-', separator, s)
    s = s.lower()
    return s

# print(decamel('HTTPError404NotFound')) # http_error_404_not_found

def sg(w, language='en', known={'aunties': 'auntie'}):
    """ Returns the singular of the given plural noun.
    """
    if w in known: 
        return known[w]
    if language == 'en':
        if re.search(r'(?i)(ss|[^s]sis|[^mnotr]us)$', w    ):  # ± 93%
            return w
        for pl, sg in (                                        # ± 98% accurate (CELEX)
          (r'          ^(son|brother|father)s-' , '\\1-'   ),
          (r'      ^(daughter|sister|mother)s-' , '\\1-'   ),
          (r'                          people$' , 'person' ),
          (r'                             men$' , 'man'    ),
          (r'                        children$' , 'child'  ),
          (r'                           geese$' , 'goose'  ),
          (r'                            feet$' , 'foot'   ),
          (r'                           teeth$' , 'tooth'  ),
          (r'                            oxen$' , 'ox'     ),
          (r'                        (l|m)ice$' , '\\1ouse'),
          (r'                        (au|eu)x$' , '\\1'    ),
          (r'                 (ap|cod|rt)ices$' , '\\1ex'  ),  # -ices
          (r'                        (tr)ices$' , '\\1ix'  ),
          (r'                     (l|n|v)ises$' , '\\1is'  ),
          (r'(cri|(i|gn|ch|ph)o|oa|the|ly)ses$' , '\\1sis' ),
          (r'                            mata$' , 'ma'     ),  # -a/ae
          (r'                              na$' , 'non'    ),
          (r'                               a$' , 'um'     ),
          (r'                               i$' , 'us'     ),
          (r'                              ae$' , 'a'      ),
          (r'           (l|ar|ea|ie|oa|oo)ves$' , '\\1f'   ),  # -ves  +1%
          (r'                     (l|n|w)ives$' , '\\1ife' ),
          (r'                 ^([^g])(oe|ie)s$' , '\\1\\2' ),  # -ies  +5%
          (r'                  (^ser|spec)ies$' , '\\1ies' ),
          (r'(eb|gp|ix|ipp|mb|ok|ov|rd|wn)ies$' , '\\1ie'  ),
          (r'                             ies$' , 'y'      ),
          (r'    ([^rw]i|[^eo]a|^a|lan|y)ches$' , '\\1che' ),  # -es   +5%
          (r'  ([^c]ho|fo|th|ph|(a|e|xc)us)es$' , '\\1e'   ),
          (r'([^o]us|ias|ss|sh|zz|ch|h|o|x)es$' , '\\1'    ),
          (r'                               s$' , ''       )): # -s    +85%
            if re.search(r'(?i)' + pl.strip(), w):
                return re.sub(r'(?i)' + pl.strip(), sg, w)
    return w                                                   #       +1%

# print(sg('avalanches')) # avalanche

def pl(w, language='en', known={}):
    """ Returns the plural of the given singular noun.
    """
    if w in known: 
        return known[w]
    if language == 'en':
        for sg, pl in (                                        # ± 98% accurate (CELEX)
          (r'                            foot$' , 'feet'   ),
          (r'                           tooth$' , 'teeth'  ),
          (r'                          person$' , 'people' ),
          (r'                      (?<!hu)man$' , 'men'    ),
          (r'                         (child)$' , '\\1ren' ),
          (r'                       (l|m)ouse$' , '\\ice'  ),
          (r'                (ch|sh|ss|s|x|z)$' , '\\1es'  ),  # -es   +4%
          (r'                            (l)f$' , '\\1ves' ),  # -ves
          (r'                        ([^f])fe$' , '\\1ves' ),  # -ves
          (r'                     ([^aeiou])y$' , '\\1ies' ),  # -ies
          (r'                           (ism)$' , '\\1'    ),
          (r'                                $' , 's'      )): # -s    +94%
            if re.search(r'(?i)' + sg.strip(), w):
                return re.sub(r'(?i)' + sg.strip(), pl, w)
    return w

# print(pl('avalanche')) # avalanches

@static(m=None)
def lang(s, confidence=0.1):
    """ Returns a (language, confidence)-dict for the given string.
    """
    if not lang.m:
        m = lang.m = Model.load(unzip(next(ls('~lang.json.zip')))) 
    else:
        m = lang.m

    v = vec(s, ('c1', 'c2')) # 92.5%
    p = m.predict(v)
    p = sparse(p, confidence)
    p = collections.Counter(p)
    return p

language = lang

# print(lang('the cat sat on the mat'))

Location = collections.namedtuple('Location', ('city', 'country', 'lat', 'lng'))

@static(m=None)
def loc(s, country=None):
    """ Returns a (Location, count)-dict of city names (EU).
    """
    if not loc.m:
        m = ls('en-loc.json')
        m = next(m)
        m = open(m, 'rb')
        m = json.load(m)
        m = m.items()
        m = {k2: Location(k2, k1, *v[k2][:2]) for k1, v in m for k2 in v} 
        m = loc.m = trie(m)
    else:
        m = loc.m

    f = m.search(s)
    f = (m.tag for m in f)
    f = (m for m in f if not country or country == m.country)
    f = collections.Counter(f)
    return f

location = loc

# for k, v in loc('Aanslag in Brussel').most_common(1):
#     print(k, v)

#---- TOKENIZER -----------------------------------------------------------------------------------
# The tokenize() function identifies tokens (= words, symbols) and sentence breaks in a string.
# The complex task involves handling abbreviations, contractions, hyphenation, emoticons, URLs, ...

EMOJI = {
    u'😊', u'☺️', u'😉', u'😌', u'😏', u'😎', u'😍', u'😘', u'😴', u'😀', u'😃', u'😄', u'😅', 
    u'😇', u'😂', u'😭', u'😢', u'😱', u'😳', u'😜', u'😛', u'😁', u'😐', u'😕', u'😧', u'😦', 
    u'😒', u'😞', u'😔', u'😫', u'😩', u'😠', u'😡', u'🙊', u'🙈', u'💔', u'❤️', u'💕', u'♥' , 
    u'👌', u'☝️', u'✌️', u'👍', u'💪', u'👊', u'👏', u'🙌', u'🙏', u'🤦', u'🤷‍♂', u'🤷', u'🌈', 
}

EMOTICON = {
    ':)', ':D', ':(', ';)', ':-)', ':P', ';-)', ':p', ':-(', ":'(", '(:', ':O', ':-D', ':o', 
    ':((', ':-P', ':-p', ':-))', '8-)', '(-:', ':-S', ';-p', ':-s', # '):', '<3', '8)'
}

_RE_EMO = '|'.join(re.escape(s) for s in EMOTICON | EMOJI)

tokens = {
    '*': [
        _RE_EMO                                                      , # :-)
        r'https?\://.*?(?=\)?[,;:!?.]*(\s|$))'                       , # http://...
        r'www\..*?\.[a-z]+'                                          , # www.goo.gl
        r'\w+\.(com?|net|org|info|be|cn|de|eu|fr|nl|ru|uk)'          , # google.com
        r'\w+\.(doc|gif|htm|jpg|mp\d|pdf|ppt|py|txt|xls|zip)'        , # cat.jpg
        r'[\w\.\-]+@\w+\.(com|net|org)'                              , # a.bc@gmail
        r'(?<![^\s])([A-Z]\.)+[A-Z]\.?'                              , # U.S.
        r'(?<![^\s])([A-Z]\.)(?= [A-Z])'                             , # J. R. R. Tolkien
        r'(?:\d+[.,:/″\'])+\d+%?'                                    , # 1.23
        r'\d+€|[$€£¥]\d+'                                            , # 100€
        r'(?<=\d)(kb|mb|gb|kg|mm|cm|km|m|ft|h|hrs|am|pm)(?=\W|$)'    , # 10 mb
        r'(?<!\w)(a\.m|ca|cfr?|etc|e\.?g|i\.?e|p\.m|P\.?S|vs|v)\.'   , # etc.
        r'(?<!\w)(Dr|dr|Mr|mr|Mrs|Ms|nr|no|Prof|prof)\.'             , # Dr.
        r'(^|\s)\d\d?\. '                                            , # 1.
        r'\.+'                                                       , #  ...
        r'[\[\(]\.+[\)\]]'                                           , # (...)
        r'[\[\(]\d+[\)\]]'                                           , # [123]
        r'p?p\.\d+'                                                  , # p.1
        r'&[a-z]+;'                                                  , # &amp;
        r'&#\d+;'                                                    , # &#38;
        r'(^|\n)>(?=[\w])'                                           , # >Hi
        r'<\w+( \w+=".*?")*( /)?>'                                   , # <p>
        r'</\w+>'                                                    , # </p>
        r'--+|==+|--*>|==*>'                                         , # =>
        r'|'.join(map(re.escape, u'….¡¿?!:;,/(){}[]\'`‘’"“”„&–—'))   , # !
    ],
    'de': [
        r'(?<!\w)(Abs|bspw|bzw|e\.V|d\.h|etw|evtl|ggf|inkl|m\.E)\.'  , # evtl.
        r'(?<!\w)(Mio|Nr|sog|u\.a|usw|v\.a|vgl|z\.B|z\.T)\.'        ], # z.B.
    'en': [
        r'(?i)(?<!\w)(Col|Gen|Rep|Rev|Sen)\.'                        , # Col.
        r'(?i)(?<=\w)\'(cause|d|ll|m|re|s|ve)'                       , # it's
        r'(?i)(?<=\w)n\'t'                                           , # isn't
        r'(?i)(?<=\w\.)\'s'                                          , # U.K.'s
        r'(?i)(?<= )(b/c|w/o|w/)(?= )'                              ], # w/
    'es': [
        r'(?i)(?<!\w)(aprox|p\.ej|Sr|Sra|Srta|Uds|Vds)\.'           ], # Sr.
    'fr': [
        r'(?i)(?<!\w)(c|d|j|l|m|n|qu|s|t|jusqu|lorsqu|puisqu)\''     , # j'ai
        r'(?i)(?<!\w)(av|ed|ex|Mme)\.'                              ], # Mme.
    'it': [
        r'(?i)\w+\'\w+'                                              , # C'è
        r'(?i)(?<!\w)(Sig|ecc)\.'                                   ], # Sig.
    'nl': [
        r'(?i)\w+\'\w+'                                              , # auto's
        r'(?i)\w\'(er|tje)s?'                                        , # 68'ers
        r'(?<!\w)(bijv|bv|d\.m\.v|e\.a|e\.d|enz|evt|excl|incl)\.'    , # enz.
        r'(?<!\w)(m\.b\.t|mevr|n\.a\.v|o\.?a|ong|t\.o\.v|zgn)\.'    ], # o.a.
    'pt': [
        r'(?i)(?<!\w)(Sr|Sra|St|Ex|ex|c)\.'                         ], # ex.
}

breaks = {
    '*': [
       (r'\r'                                      , '\n'           ), # \n
       (r' " (.*?) " '                             , ' "< \\1 >" '  ), # "< Oh >"
       (r" ' (.*?) ' "                             , " '< \\1 >' "  ), # '< Oh >'
       (u' (…|\.+) (?=[A-Z])'                      , ' \\1 \n '     ), #  Oh… \n Wow
       (r' ([.!?]) (?=[^.!?])'                     , ' \\1 \n '     ), #  Oh! \n Wow
       (u' ([.!?]) \n (>"|>\'|”|’)'                , ' \\1 \\2 \n'  ), # “Oh!”\n Wow
       (u' ([!?]) \n ([–—] [a-z])'                 , ' \\1 \\2'     ), # –Oh!–
       (u' (>"|>\'|”|’) ([A-Z])'                   , ' \\1 \n \\2'  ), # “Oh!”\n They
       (u' (>"|>\'|”|’) \n ([(a-z])'               , ' \\1 \\2'     ), # “Oh!” they said
       (r' "< ((?:[\w,.!?]+ ){1,5})\n ' \
            r'((?:[\w,.!?]+ ){1,5})>" '            , ' "< \\1\\2>" '), # “Oh! Wow!”
       (r' "< ((?:[\w,.!?]+ ){1,5})\n ([a-z])'     , ' "< \\1\\2'   ), # “Oh! wow [...]”
       (r' \n , '                                  , ' , '          ), #  Oh!, they said
       (r' \n ((#\w+)( #\w+)*)( |$)'               , ' \\1 \n'      ), #  Oh! #Wow \n
       (r' \n (((%s) ?)+)( |$)'          % _RE_EMO , ' \\1 \n'      ), #  Oh! :) \n
       (r' (%s) (?=[A-Z])'               % _RE_EMO , ' \\1 \n'      ), #  Oh :-) \n He
       (r' ((etc|enz|usw)\.) (?=[A-Z])'            , ' \\1 \n'      ), # etc. The
       (r'(?<=[a-z])- *\n *([a-z])'                , '\\1'          ), # be-\ncause
       (r' ([\'"])< '                              , ' \\1 '        ), # "
       (r' >([\'"]) '                              , ' \\1 '        ), # '
       (r'\s*\n\s*'                                , '\n'           ), # \n\n
    ]
}

# Token splitting rules:
# A token can be a string of alphanumeric characters, i.e., a "word".
# A token can be a URL, domain or file name, email, without trailing commas.
# A token can be a number, including decimals, thousands and currency signs.
# A token can be a known (language-specific) abbreviation or contraction.
# A token can be a named entity, emoticon, or a HTML tag with attributes.

# Sentence break rules:
# Sentence breaks are marked by .!? or by … if followed by a capital letter.
# Sentence breaks in quotes are ignored if not followed by a capital letter.
# Sentence breaks can be followed by trailing quotes, hashtags or emoticons.

def tokenize(s, language='en', known=[], separator=' '): # ~125K tokens/sec
    """ Returns the string with punctuation marks split from words ,
        and sentences separated by newlines.
    """
    f = ['' for ch in s] # token flags
    r = []
    r.extend(known)
    r.extend(tokens.get(language, ()))
    r.extend(tokens.get('*'))
    # Scan tokens:
    for r in r:
        for m in re.finditer(r, s, flags=re.U):
            i = m.start()
            j = m.end() - 1
            if not f[i] and i == j:
                f[i] = '^$'
            if not f[i]: # token start
                f[i] = '^'
            if not f[j]: # token end
                f[j] = '$'
            for i in range(i+1, j):
                f[i] = '.'
    a = []
    # Split tokens:
    for ch, f in zip(s, f):
        if '^' in f:
            ch  = separator + ch
        if '$' in f:
            ch += separator
        a.append(ch)
    s = ''.join(a)
    s = re.sub('%s+' % separator, separator, s)
    # Split breaks:
    for a, b in breaks.get('*') + breaks.get(language, []):
        s = re.sub(a, b, s)
    return s.strip()

tok = tokenize

# s = u"RT @user: Check it out... 😎 (https://www.textgain.com) #Textgain cät.jpg"
# s = u"There's a sentence on each line, each a space-separated string of tokens (i.e., words). :)"
# s = u"Title\nA sentence.Another sentence. ‘A citation.’ By T. De Smedt."
# 
# print(tok(s))

def scan(s, language='en', known=[]):
    """ Returns a list of token breakpoints,
        which can be used for de-tokenizing.
    """
    b = u'␣'
    s = s.replace(b, '_')
    s = re.sub(r'(^\s)', '_', s)
    s = tokenize(s, language, known, b)
    s = re.sub(' %s' % b, ' ', s)
    s = re.sub('%s ' % b, ' ', s)
    s = s.strip(b)
    a = []
    for i, ch in enumerate(s):
        if ch == b:
            a.append(i - len(a))
    return a

# print(scan(u'Here, kitty kitty!'))

def wc(s):
    """ Returns a (word, count)-dict, lowercase.
    """
    s = s.lower()
    s = s.split()
    f = collections.Counter(s)
    return f

# print(wc(tok('The cat sat on the mat.')))

#---- PART-OF-SPEECH TAGGER -----------------------------------------------------------------------
# Part-of-speech tagging predicts the role of each word in a sentence: noun, verb, adjective, ...
# Different words have different roles according to their position in the sentence (context).
# In 'Can I have that can of soda?', 'can' is used as a verb ('I can') and a noun ('can of').

# We want to train a machine learning algorithm (Percepton) with context vectors as examples,
# where each context vector includes the word, word suffix, and words to the left and right.

# The part-of-speech tag of unknown words is predicted by their suffix (e.g., -ing, -ly)
# and their context (e.g., a determiner is often followed by an adjective or a noun).

def ctx(*w):
    """ Returns the given (token, tag) list parameters as 
        a context vector (i.e., token, tokens left/right) 
        that can be used to train a part-of-speech tagger.
    """
    m = len(w) // 2 # middle
    v = dict()
    for i, (w, tag) in enumerate(w):
        i -= m
        if i == 0:
            v.update({
                ' '                                : 1, # bias
                '- %+d %i' % (i, '-' in w        ) : 1, # hyphenation
                '. %+d %i' % (i,  not w.isalpha()) : 1, # punctuation
                '! %+d %i' % (i,      w.isupper()) : 1, # capitalization
                '@ %+d %i' % (i, w[:1 ].isupper()) : 1, # capitalized?
                '# %+d %s' % (i, w[:1 ].lower()  ) : 1, # token head (1)
                '^ %+d %s' % (i, w[:3 ].lower()  ) : 1, # token head (3)
                '* %+d %s' % (i, w[-6:].lower()  ) : 1, # token
                '$ %+d %s' % (i, w[-3:].lower()  ) : 1, # token tail (3)
                's %+d %s' % (i, w[-1:].lower()  ) : 1, # token tail (1)
            })
        else:
            v.update({
                '# %+d %s' % (i, w[:1 ].lower()  ) : 1, # token head (1)
                's %+d %s' % (i, w[-1:].lower()  ) : 1, # token tail (1)
                '$ %+d %s' % (i, w[-3:].lower()  ) : 1, # token tail (3)
                '? %+d %s' % (i, tag             ) : 1, # token tag L/R
            })
    return v

# print(ctx(('The', 'DET'), ('cat', 'NOUN'), ('sat', 'VERB'))) # context of 'cat'
#
# {
#     ' '        , 
#     '$ -1 The' , 
#     '? -1 DET' , 
#     '- +0 0'   , 
#     '. +0 0'   , 
#     '! +0 0'   , 
#     '@ +0 0'   , 
#     '# +0 c'   , 
#     '^ +0 cat' , 
#     '* +0 cat' , 
#     '$ +0 cat' , 
#     '$ +1 sat' , 
#     '? +1 VERB', 
# }

@printable
class Tagged(list):

    def __init__(self, s=''):
        """ Returns the tagged string as a list of (token, tag)-tuples.
        """
        if isinstance(s, list):
            list.__init__(self, ((u(w), tag) for w, tag in s))
        if isinstance(s, (str, unicode)) and s:
            for w in s.split():
                w = u(w)
                w = w.rsplit('/', 1)
                w = tuple(w)
                self.append(w)

    def __str__(self):
        return ' '.join('/'.join(w) for w in self)

    def __repr__(self):
        return 'Tagged(%s)' % repr(u(self))

# s = 'The/DET cat/NOUN sat/VERB on/PREP the/DET mat/NOUN ./PUNC'
# for w, tag in Tagged(s):
#     print(w, tag)

tagger = LazyDict() # {'en': Model}

for f in ls('*-pos.json.zip'):
    tagger[f[-15:-13]] = lambda f=f: Perceptron.load(unzip(f))
#                                ^ early binding

lexicon = {
    'en': {
        '.'    : 'PUNC', # top 20
        ','    : 'PUNC',
        'a'    : 'DET' ,
        'an'   : 'DET' ,
        'the'  : 'DET' ,
        'I'    : 'PRON',
        'it'   : 'PRON',
        'is'   : 'VERB',
        'be'   : 'VERB',
        'was'  : 'VERB',
        'has'  : 'VERB',
        'have' : 'VERB',
        'from' : 'PREP',
        'for'  : 'PREP',
        'in'   : 'PREP',
        'to'   : 'PREP',
        'of'   : 'PREP',
        'or'   : 'CONJ',
        'and'  : 'CONJ',
    }
}

lexical = [
    (r'^[\W_]{4,}' , 'PUNC'), # /*----*/
    (r'^[\d.,]+'   , 'NUM' ), # 99,999.9
    (r'^@[\w_]+'   , 'NAME'), # @xyz
    (r'@\w+\.\w+$' , 'NAME'), # @xyx.com
    (r'^https?://' , 'URL' ), # https://
    (r'^www\..'    , 'URL' ), # www.xyz
    (r'^#\w+'      , '#'   ), # #xyz
    (r'^:-?[()Dpo]', ':)'  ), # :-)
]

def tag(s, language='en'): # ~5K tokens/sec
    """ Returns the tagged string as a list of (token, tag)-tuples.
    """
    f = lexicon.get(language, {})
    m = tagger[language]
    a = s.split()
    a.insert(0, '')
    a.insert(0, '')
    a.append('')
    a.append('')
    a = [[w, ''] for w in a]
    for w in nwise(a, n=5):
        try:
            tag = f[w[2][0]]              # lexicon word
        except:
            for r, tag in lexical:        # lexical rule
                if re.search(r, w[2][0]):
                    break
            else:
                tag = m.predict(ctx(*w))  # {tag: probability}
                tag = top(tag)            # (tag, probability)
                tag = tag[0]
        w[2][1] = tag
    a = a[2:-2]
    a = map(tuple, a)
    a = list(a)
    a = Tagged(a)
    return a

def parse(s, language='en', tokenize=tokenize):
    """ Returns the tagged string as an iterator of sentences,
        where each sentence is a list of (token, tag)-tuples.
    """
    s = tokenize(s, language)
    s = s.split('\n')
    for s in s:
        yield tag(s, language)

# for s in parse("We're all mad here. I'm mad. You're mad."):
#     print(repr(s))

PTB = {       # Penn Treebank tagset                                          # EN
    u'NOUN' : {'NN', 'NNS', 'NNP', 'NNPS', 'NP'},                             # 30%
    u'VERB' : {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD'},                # 14%
    u'PUNC' : {'LS', 'SYM', '.', ',', ':', '(', ')', '``', "''", '$', '#'},   # 11%
    u'PREP' : {'IN', 'PP'},                                                   # 10%
    u'DET'  : {'DT', 'PDT', 'WDT', 'EX'},                                     #  9%
    u'ADJ'  : {'JJ', 'JJR', 'JJS'},                                           #  7%
    u'ADV'  : {'RB', 'RBR', 'RBS', 'WRB'},                                    #  4%
    u'NUM'  : {'CD', 'NO'},                                                   #  4%
    u'PRON' : {'PR', 'PRP', 'PRP$', 'WP', 'WP$'},                             #  3%
    u'CONJ' : {'CC', 'CJ'},                                                   #  2%
    u'X'    : {'FW'},                                                         #  2%
    u'PRT'  : {'POS', 'PT', 'RP', 'TO'},                                      #  2%
    u'INTJ' : {'UH'},                                                         #  1%
}

WEB = dict(PTB, **{
    u'NOUN' : {'NN', 'NNS', 'NP'},                                            # 14%
    u'NAME' : {'NNP', 'NNPS', '@'},                                           # 11%
    u'PRON' : {'PR', 'PRP', 'PRP$', 'WP', 'WP$', 'PR|MD', 'PR|VB', 'WP|VB'},  #  9%
    u'URL'  : {'URL'},                         # 'youve'  'thats'  'whats'    #  1%
    u':)'   : {':)'},                                                         #  1%
    u'#'    : {'#'},                                                          #  1%
})
WEB['PUNC'].remove('#')
WEB['PUNC'].add('RT')

def universal(w, tag, tagset=WEB):
    """ Returns a simplified tag (e.g., NOUN) for the given Penn Treebank tag (e.g, NNS).
    """
    tag = tag.split('-')[0]
    tag = tag.split('|')[0]
    tag = tag.split('&')[0]
    if tag == w == '#':
        return w, 'PUNC' # != hashtag
    for k in tagset:
        if tag in tagset[k]:
            return w, k
    return w, tag

# print(universal('rabbits', 'NNS'))

# The 1989 Wall Street Journal corpus contains 1 million manually annotated words, e.g.:
# Pierre/NNP Vinken/NNP ,/, 61/CD years/NNS old/JJ ,/, will/MD join/VB the/DT board/NN 
# as/IN a/DT nonexecutive/JJ director/NN Nov./NNP 29/CD ./.

# WSJ = u(open(cd('/corpora/wsj.txt')).read())
# WSJ = WSJ.split('\n')
# 
# data = []
# for s in WSJ:
#     for w in nwise(Tagged('/ / %s / /' % s), n=5):
#         w = [universal(*w) for w in w]
#         data.append((ctx(*w), w[2][1]))
# 
# print(kfoldcv(Perceptron, data, k=3, n=5, weighted=True, debug=True)) # 0.96 0.96
# 
# en = Perceptron(data, n=5)
# en.save(open('en.json', 'w'))
# 
# print(tag('What a great day! I love it.'))

#---- PART-OF-SPEECH CHUNKER ----------------------------------------------------------------------
# The chunk() function yields parts of a part-of-speech-tagged sentence that match a given pattern.
# For example, 'ADJ* NOUN' yields all nouns in a sentence, and optionally the preceding adjectives.

# The constituents() function yields NP, VP, AP and PP phrases.
# A NP (noun phrase) is a noun + preceding determiners and adjectives (e.g., 'the big black cat').
# A VP (verb phrase) is a verb + preceding auxillary verbs (e.g., 'might be doing'). 

TAG = {
    'NAME' ,
    'NOUN' ,
    'VERB' ,
    'PUNC' ,
    'PREP' ,
    'DET'  ,
    'ADJ'  ,
    'ADV'  ,
    'NUM'  ,
    'PRON' ,
    'CONJ' ,
    'X'    ,
    'PRT'  ,
    'INTJ' ,
    'URL'  ,
    ':)'   ,
    '#'
}

inflections = {
    'en': {
        'aux'  : r"can|shall|will|may|must|could|should|would|might|'ll|'d",
        'be'   : r"be|am|are|is|was|were|being|been|'m|'re|'s",
        'have' : r"have|has|had|having|'ve",
        'do'   : r"do|does|did|doing"
    }
}

class Phrase(Tagged):

    def __repr__(self):
        return 'Phrase(%s)' % repr(u(self))

_RE_TAG = '|'.join(map(re.escape, TAG)) # NAME|NOUN|\:\)|...

def chunk(pattern, text, replace=[], language='en'):
    """ Returns an iterator of matching Phrase objects in the given tagged text.
        The search pattern is a sequence of tokens (laugh-, -ing), tags (VERB), 
        token/tags (-ing/VERB), escaped characters (:\)) or control characters: 
        - ^ $ begin/end
        - ( ) group
        -  |  options: NOUN|PRON CONJ|,
        -  *  0+ tags: NOUN*
        -  +  1+ tags: NOUN+
        -  ?  <2 tags: NOUN?
    """
    def is_tag(w):
        return any(re.match(w+'$', t) for t in TAG)         # 'AD[JV]'
    def is_tags(w):
        return all(is_tag(w) for w in w.split('|'))         # 'ADJ|ADV'

    s = re.sub(r'\s+', ' ', ' %s ' % pattern)
    s = re.sub(r'(?<!\\)([\^])' , ' \\1 ', s)               # '^ADJ' => '^ ADJ'
    s = re.sub(r' ([*+?])(?=[ $])', ' -\\1', s)             # ':) ?' => ':) -?'
    s = re.sub(r'(?<!\\)([()$])', ' \\1 ', s)               # 'ADJ$' => 'ADJ $'
    s = s.strip()
    p = []

    for w in s.split(' '):
        if w in ('(', ')', '^', '$', '*', '+', '?', ''):
            p.append(w)
            continue
        for k, v in replace:
            w = w.replace(k.upper(), v)
        for k, v in inflections.get(language, {}).items():
            w = w.replace(k.upper(), v)                     # 'BE' => 'be|am|are|is'

        try:
            w, x, _ = re.split(r'(?<!\\)([*+?])$', w)       # ['ADJ|-ing', '+']
        except ValueError:
            x = ''
        if not re.search(r'(?<!\\)\/', w):
            a = re.split(r'(?<!\\)\|', w)                   # ['ADJ', '-ing']
            for i, w in enumerate(a):
                if is_tag(w):
                    a[i] = r'(?:\S+/%s)' % w                # '(?:\S+/ADJ)'
                else:
                    a[i] = r'(?:%s/(?:%s))' % (w, _RE_TAG)  # '(?:-ing/[A-Z]+)'
            w = '|'.join(a)                                 # '(?:\S+/ADJ)|(?:-ing/[A-Z]+)'
        elif is_tags(w.split('/')[-1]):
            w = re.sub(r'(?<!\\)/(?!.*/)', ')/(?:', w)      # '(?:-ing)/(?:VERB|ADJ)'
        else:
            w = '%s/(?:%s)' % (w, _RE_TAG)                  # '(?:1\/2)/[A-Z]+'

        w = '(?:%s)' % w
        w = '(?:%s )%s' % (w, x)                            # '(?:(?:-ing/[A-Z]+) )+'
        w = re.sub(r'\(\?:-', r'(?:\\S*', w)                #     '\S*ing/[A-Z]+'
        w = re.sub(r'\(\?:(\S*)-/', r'(?:\\b\1\\S*', w)
        w = re.sub(r'\/', '\\\/', w)
        p.append(w)

    p = '(%s)' % ''.join(p)
    for m in re.finditer(p, '%s ' % text, re.I):
        m = (m.strip() for m in m.groups() if m)
        m = map(Phrase, m)
        m = tuple(m)
        if len(m) == 1:
            yield m[0]
        if len(m) >= 2:
            yield m

# for m in \
#   chunk('ADJ', 
#    tag(tok('A big, black cat.'))):
#     print(u(m))

# for m, g1, g2 in \
#   chunk('DET? (NOUN) AUX? BE (-ry)', 
#    tag(tok("The cats'll be hungry."))): 
#     print(u(g1), u(g2))

# for m, g1, g2, g3 in \
#   chunk('(NOUN|PRON) BE ADV? (ADJ) than (NOUN|PRON)', 
#    tag(tok("Cats are more stubborn than dogs."))):
#     print(u(g1), u(g2), u(g3))

def constituents(text, language='en'):
    """ Returns an iterator of (Phrase, tag)-tuples in the given tagged text,
        with tags NP (noun phrase), VP (verb phrase), AP (adjective phrase) 
        or PP (prepositional phrase).
    """
    if language in ('da', 'de', 'en', 'nl', 'no', 'sv'): # Germanic
        P = (('NP' , 'DET|PRON? NUM* (ADV|ADJ+ CONJ|, ADV|ADJ)* ADV|ADJ* -ing/VERB* NOUN|NAME+'),
             ('NP' , 'NOUN|NAME DET NOUN|NAME'),
             ('NP' , 'PRON'),
             ('AP' , '(ADV|ADJ+ CONJ|, ADV|ADJ)* ADV* ADJ+'),
             ('VP' , 'PRT|ADV* VERB+ ADV?'),
             ('PP' , 'PREP+'),
             (  '' , '-')
        )
    s = u(text)
    s = re.sub(r'\s+', ' ', s)
    while s:
        for tag, p in P:
            try:
                m = next(chunk('^(%s)' % p, s, language=language))[0]; break
            except StopIteration:
                m = ''
        if not m:
            m = Phrase(s.split(' ', 1)[0])
        if not m:
            break
        s = s[len(u(m)):]
        s = s.strip()
        yield m, tag

phrases = constituents

# s = parse('The black cat is dozing lazily on the couch.')
# for phrase, tag in constituents(next(s)):
#     print(tag, u(phrase))

def head(phrase, tag='NP', language='en'):
    """ Returns the head-word in the given phrase (naive).
    """
    if language in ('da', 'de', 'en', 'nl', 'no', 'sv') and tag == 'NP' or tag == 'VP':
        phrase = reversed(phrase) # cat fight <=> combat de chats
    for w, pos in phrase:
        if pos == 'NOUN' and tag == 'NP' \
        or pos == 'VERB' and tag == 'VP' \
        or pos ==  'ADJ' and tag == 'AP':
            return w

# print(head(Phrase('cat/NOUN fight/NOUN')))
# print(head(Phrase('combat/NOUN de/PREP chats/NOUN'), language='fr'))

#---- KEYWORD EXTRACTION --------------------------------------------------------------------------
# Keyword extraction aims to determine the topic of a text.

df = LazyDict() # document frequency

for f in ls('*-doc.json'):
    df[f[-11:-9]] = lambda f=f: json.load(open(f, 'rb'))

def keywords(s, language='en', n=10):
    """ Returns a list of n keywords.
    """
    s = tok(s, language)                  # "Boo Boo needs chow"
    s = tag(s, language)                  # "Boo/NAME Boo/NAME needs/VERB chow/NOUN"
    s = chunk('NAME+', s), \
        chunk('NOUN+', s)
    s = itertools.chain(*s)               # ['Boo/NAME Boo/NAME', 'chow/NOUN']
    s = [(w for w, pos in p) for p in s]  # [['Boo', 'Boo'], ['chow']]
    s = [' '.join(p).lower() for p in s]  # ['boo boo', 'chow']
    f = zip(s, range(1, 1 + len(s)))      # [('boo boo', 1), ('chow', 2)]
    f = list(f)
    f = reversed(f)
    f = dict({k: 1.0 / v for k, v in f})  # {'boo boo': 1.0, 'chow': 0.5}
    f = collections.Counter(s, **f)       # {'boo boo': 2.0, 'chow': 1.5} (tf)
    f = f.most_common(n)
    return [k for k, v in f]

kw = keywords

# print(kw('Boo Boo needs chow.'))

#---- SENTIMENT ANALYSIS --------------------------------------------------------------------------
# Sentiment analysis aims to determine the affective state of (subjective) text,
# for example whether a customer review is positive or negative about a product.

negation = {
    'en': { 'no', 'not', "n't",            },
    'es': { 'no', 'nunca',                 },
    'de': { 'kein', 'keine', 'nicht'       },
    'fr': { 'aucun', 'aucune', 'ne', 'pas' },
    'nl': { 'geen', 'niet', 'nooit'        },
}

intensifiers = {
    'en': { 'very', 'really', 'big'        },
    'es': { 'muy', 'mucho',                },
    'de': { 'sehr',                        },
    'fr': {u'très', 'vraiment',            },
    'nl': { 'echt', 'erg', 'heel', 'zeer'  },
}

polarity = {
    'en': {
        'awesome' : +1.0,
        'good'    : +0.5,
        'bad'     : -0.5,
        'awful'   : -1.0,
    }
}

polarity = LazyDict()

for f in ls('*-pov.json'): # point-of-view
    polarity[f[-11:-9]] = lambda f=f: json.load(open(f, 'rb'))

def sentiment(s, language='en'):
    """ Returns the polarity of the string as a float,
        from negative (-1.0) to positive (+1.0).
    """
    p = polarity.get(language, {})
    s = u(s)
    s = s.lower()
    s = s.split()
    a = []
    i = 0
    while i < len(s):
        for n in (2, 1,): # n-grams
            w = s[i:i+n]
            w = ' '.join(w)
            if w in p:
                v = p[w]
                if i > 0:
                    if s[i-1] in negation.get(language, ()):     # "not good"
                        v *= -1.0
                    if s[i-1] in intensifiers.get(language, ()): # "very bad"
                        v *= +2.0
                    if s[i-1] in p and v != p[w]:
                        a.pop()
                a.append(v)
                i += n - 1
                break
        i += 1
    v = avg(a)
    v = max(v, -1.0)
    v = min(v, +1.0)
    return v

pov = sentiment

# s = 'This movie is very bad!'
# s = tokenize(s)
# s = sentiment(s)
# print(s)

def positive(s, language='en', threshold=0.1):
    # = 75% (Pang & Lee polarity dataset v1.0)
    return sentiment(s, language) >= threshold

#---- WORD EMBEDDING ------------------------------------------------------------------------------
# Word embedding is a combination of context vectors and dimensionality reduction.
# For each word in a large amount of text, the context vector represents a word's
# preceding and successive words (context). Distance between vectors = semantics.
# Context vectors can have many features, so dimension reduction is used to scale
# down the model in memory size and computation time, while preserving distances.

def cooccurrence(s, window=2, ignore=set()):
    """ Returns a dict of (word, context)-items, where context is 
        a dict of preceding and successive words and their counts.
    """
    f = collections.defaultdict(collections.Counter) # {word1: {word2: count}}
    i = window
    s = s.split()
    s = [None] * i + s
    s+= [None] * i

    for w in ngrams(s, i * 2 + 1):
        for j in range(len(w)):
            if w[i] in ignore:
                break
            if w[j] in ignore:
                continue
            if w[j] is None:
                continue
            if i != j:
                f[w[i]][w[j]] += 1
    return f

cooc = cooccurrence

# print(cooc('the cat sat on the mat', 1))['the']

class Embedding(collections.OrderedDict):

    def __init__(self, s, m=10000, n=100, reduce=lambda v, n: [v.most_common(n) for v in v], **kwargs):
        """ Returns a dict of (word, context)-items, where context is 
            a dict of preceding and successive words and their counts.
        """
        collections.OrderedDict.__init__(self)

        if s:
            a = cooc(s, **kwargs) # {word1: {word2: count}}
            a = a.items()
            a = sorted(a, key=lambda wv: sum(wv[1].values()))
            a = reversed(a)
            a = list(a)[:m]       # most frequent words
            w, v = zip(*a)
            v = reduce(v, n)      # dimension reduction
            v = map(dict, v)
            a = zip(w, v)
            self.update(a)

    def similar(self, w, n=10, distance=cos):
        """ Returns a list of n (similarity, word)-tuples.
        """
        if not isinstance(w, dict):
            v1 = self.get(w, {})
        else:
            v1 = w
        q = ((1 - distance(v1, v2), w2) for w2, v2 in self.items())
        q = heapq.nlargest(n, q)
        q = [(w, v) for w, v in q if w > 0]
        return q

    def vec(self, s):
        """ Returns a vector for the given word or sentence.
        """
        return centroid(self[w] for w in s.split() if w in self)

    def save(self, f, format='json'):

        if format == 'json':
            f.write(json.dumps(self))

        if format == 'txt':
            h = features(self.values())
            h = sorted(h)
            s = ''
            for w, v in self.items():
                s += '\n%s ' % w
                s += ' '.join('%.5f' % v.get(i, 0) for i in h)
            f.write(s.strip())

    @classmethod
    def load(cls, f, format='json'):
        e = cls('')

        if format == 'json':
            e.update(json.loads(f.read()))

        if format == 'txt':      # GloVe
            for s in f:          # word1 0.0 0.0 0.0 ...
                s = u(s)
                s = s.split(' ') # word2 0.0 0.0 0.0 ...
                w = s[0]
                v = s[1:]
                v = map(float, v)
                v = {i: n for i, n in enumerate(v) if n}
                e[w] = v

        return e

Embeddings = Embedding

def add(v1, v2):
    return {f: v1.get(f, 0) + v2.get(f, 0) for f in features((v1, v2))}

def sub(v1, v2):
    return {f: v1.get(f, 0) - v2.get(f, 0) for f in features((v1, v2))}

# e = open('glove.6B.50d.txt')
# e = sliced(e, 0, 100000)
# e = Embedding.load(e, 'txt')

# print(e.similar('king', 10)) # = president, emperor, leader
# print(e.similar(add(sub(e['king'], e['man']), e['woman']))) # = queen

#---- WORDNET -------------------------------------------------------------------------------------
# WordNet is a free lexical database of synonym sets, and relations between synonym sets.

SYNSET = r'^\d{8} \d{2} \w .. ((?:.+? . )+?)\d{3} ((?:..? \d{8} \w .... )*)(.*?)\| (.*)$'
# '05194874 07 n 02 grip 0 grasp 0 001 @ 05194151 n 0000 | an intellectual understanding'
#  https://wordnet.princeton.edu/wordnet/man/wndb.5WN.html#sect3

POINTER = {
    'antonym'  : '!',  # 
    'hypernym' : '@',  # grape -> fruit
    'hyponym'  : '~',  # grape -> muscadine
    'holonym'  : '#',  # grape -> grapevine
    'meronym'  : '%',  # grape -> wine
}

class WordNet(dict):

    def __init__(self, path='WordNet-3.0'):
        """ Opens the WordNet database from the given path 
            (that contains dict/index.noun, dict/data.noun, ...)
        """
        self._f = {} # {'n': <open file 'dict/index.noun'>}

        for k, v in (('n', 'noun'), ('v', 'verb'), ('a', 'adj' ), ('r', 'adv' )):

            f = cd(path, 'dict',  'data.%s' % v)
            f = open(f, 'rb')
            self._f[k] = f

            f = cd(path, 'dict', 'index.%s' % v)
            f = open(f, 'r')
            for s in f:
                if not s.startswith(' '):
                    s = s.strip()
                    s = s.split(' ')
                    p = s[-int(s[2]):]
                    w = s[0]
                    w = w.replace('_', ' ')
                    self[w, k] = p # {('grasp', 'n'): (offset1, ...)}
            f.close()

    def synset(self, offset, pos='n'):
        f = self._f[pos]
        f.seek(int(offset))
        s = f.readline()
        s = s.strip()
        s = s.decode('utf-8')
        m = re.match(SYNSET, s)
        w = m.group(1)
        p = m.group(2)
        g = m.group(4)
        p = tuple(chunks(p.split(' '), n=4))  # (pointer, offset, pos, source/target)
        w = tuple(chunks(w.split(' '), n=2))  # (word, sense)
        w = tuple(w.replace('_', ' ') for w, i in w)

        return Synset(offset, pos, lemma=w, pointers=p, gloss=g, factory=self.synset)

    def synsets(self, w, pos='n'):
        """ Returns a tuple of senses for the given word,
            where each sense is a Synset (= synonym set).
        """
        w = w.lower()
        w = w, pos
        return tuple(self.synset(offset, pos) for offset in self.get(w, ()))

    def __call__(self, *args, **kwargs):
        return self.synsets(*args, **kwargs)

Wordnet = WordNet

class Synset(tuple):

    def __new__ (self, offset, pos, lemma, pointers=[], gloss='', factory=None):
        return tuple.__new__(self, lemma)

    def __init__(self, offset, pos, lemma, pointers=[], gloss='', factory=None):
        """ A set of synonyms, with semantic relations and a definition (gloss).
        """
        self.synset   = factory
        self.offset   = offset
        self.pos      = pos
        self.pointers = pointers
        self.gloss    = gloss

    @property
    def id(self):
        return '%s-%s' % (self.offset, self.pos)

    # Synset.hypernyms, .hyponyms, ...
    def __getattr__(self, k):
        v = POINTER[k.replace('_', ' ')[:-1]] # -s
        v = tuple(self.synset(p[1], p[2]) for p in self.pointers if p[0].startswith(v))
        setattr(self, k, v) # lazy
        return v

    def __repr__(self):
        return 'Synset(%s)' % tuple.__repr__(self)

# wn = Wordnet(path='WordNet-3.0')
# for s in wn.synsets('grasp', pos='n'):
#     print(s)
#     print(s.gloss)
#     print(s.hyponyms)
#     print()

##### WWW #########################################################################################

#---- OAUTH ---------------------------------------------------------------------------------------
# The Open standard for Authorization (OAuth) is used to encrypt requests, for example by Twitter.
# The protocol is documented on https://tools.ietf.org/html/rfc5849. Do not change the code below.

def oauth(url, data={}, key='', token='', secret=('',''), method='GET'):
    """ Returns (url, data), where data is updated with OAuth 1.0 authorization.
    """

    def nonce():
        return hashlib.md5(b('%s%s' % (time.time(), random.random()))).hexdigest()

    def timestamp():
        return int(time.time())

    def encode(s):
        return urllib.parse.quote(b(s), safe='~')

    def hash(s, key):
        return hmac.new(b(s), b(key), hashlib.sha1).digest()

    def base(url, data={}, method='GET'):
        # https://tools.ietf.org/html/rfc5849#section-3.4.1
        s  = encode(method.upper())  + '&'
        s += encode(url.rstrip('?')) + '&'
        s += encode('&'.join('%s=%s' % (
             encode(k), 
             encode(v)) for k, v in sorted(data.items())))
        return s

    def sign(url, data={}, method='GET', secret=('','')):
        # https://tools.ietf.org/html/rfc5849#section-3.4.2
        s  = encode(secret[0]) + '&' 
        s += encode(secret[1])
        s  = hash(s, base(url, data, method))
        s  = base64.b64encode(s)
        return s

    data = dict(data, **{
        'oauth_nonce'            : nonce(),
        'oauth_timestamp'        : timestamp(),
        'oauth_consumer_key'     : key,
        'oauth_token'            : token,
        'oauth_signature_method' : 'HMAC-SHA1',
        'oauth_version'          : '1.0',
    })

    data['oauth_signature'] = sign(url.split('?')[0], data, method.upper(), secret)

    return url, data

OAuth = collections.namedtuple('Oauth', ('key', 'token', 'secret'))

#---- SERIALIZATION -------------------------------------------------------------------------------

def serialize(url='', data={}):
    """ Returns a URL with a query string of the given data.
    """
    p = urllib.parse.urlsplit(url)
    q = urllib.parse.parse_qsl(p.query)
    q.extend((b(k), b(v)) for k, v in sorted(data.items()))
    q = urllib.parse.urlencode(q, doseq=True)
    p = p.scheme, p.netloc, p.path, q, p.fragment
    s = urllib.parse.urlunsplit(p)
    s = s.lstrip('?')
    return s

# print(serialize('http://www.google.com', {'q': 'cats'})) # http://www.google.com?q=cats

#---- REQUESTS & STREAMS --------------------------------------------------------------------------
# The download(url) function returns the HTML (JSON, image data, ...) at the given url.
# If this fails it will raise NotFound (404), Forbidden (403) or TooManyRequests (429).

class BadGateway      (Exception): pass # 502
class BadRequest      (Exception): pass # 400
class Forbidden       (Exception): pass # 403
class NotFound        (Exception): pass # 404
class TooManyRequests (Exception): pass # 429
class Timeout         (Exception): pass

cookies = cookielib.CookieJar()
context = None # ssl.SSLContext
proxies = {}   # {'https': ...}

def request(url, data={}, headers={}, timeout=10):
    """ Returns a file-like object from the given URL.
    """
    if isinstance(data, dict):
        if headers.get('Content-Type') == 'application/json':
            data = b(json.dumps(data))
        else:
            data = b(urllib.parse.urlencode(data))

    f = []
    if cookies:
        f.append(urllib.request.HTTPCookieProcessor(cookies))
    if context:
        f.append(urllib.request.HTTPSHandler(context=context))
    if proxies:
        f.append(urllib.request.ProxyHandler(proxies=proxies))
    try:
        q = urllib.request.Request(url, data or None, headers)
        f = urllib.request.build_opener(*f)
        f = f.open(q, timeout=timeout)

    except urllib.error.URLError as e:
      # print(e.read())
        status = getattr(e, 'code', None) # HTTPError
        if status == 404:
            raise NotFound
        if status == 403:
            raise Forbidden
        if status == 401:
            raise Forbidden
        if status == 400:
            raise BadRequest
        if status == 402:
            raise TooManyRequests
        if status == 420:
            raise TooManyRequests
        if status == 429:
            raise TooManyRequests
        if status == 502:
            raise BadGateway
        raise e

    except socket.error as e:
        if 'timed out' in repr(e.args):
            raise Timeout
        else:
            raise e

    log.info(url)
    return f

CACHE = cd('cache')

@static(ready=0)
def download(url, data={}, headers={}, timeout=10, delay=0, cached=False):
    """ Returns the content at the given URL, as a byte string.
    """
    k = re.sub(r'&?oauth_[\w=%-]+', '', url)
    k = hashlib.sha1(b(k)).hexdigest()[:20]
    k = os.path.join(CACHE, '%s.txt' % k)

    if not os.path.exists(CACHE):
        os.makedirs(CACHE)
    if not os.path.exists(k) \
    or not cached:
        s = request(url, data, headers, timeout)
        s = s.read()
        if cached:
            with open(k, 'wb') as f:
                f.write(s)

    # Rate limiting:
    time.sleep(max(0, download.ready - time.time()))
    download.ready = time.time() + delay

    if not cached:
        return s
    with open(k, 'rb') as f:
        return f.read()

dl = download

# print(u(download('http://textgain.com')))

class stream(list):

    def __init__(self, request, bytes=1024):
        """ Returns an iterator of read data for the given request().
        """
        self.request = request
        self.bytes = bytes

    def __iter__(self):
        b = b'' # buffer
        while 1:
            try:
                b = b + self.request.read(self.bytes)
                b = b.split(b'\n')
                for s in b[:-1]:
                    if s.strip(): 
                        yield s
                b = b[-1]

            except socket.error as e:
                if 'timed out' in repr(e.args):
                    raise Timeout
                else:
                    raise e

def redirect(url, *args, **kwargs):
    """ Returns the redirected URL.
    """
    return request(url, *args, **kwargs).geturl()

def headers(url, *args, **kwargs):
    """ Returns a headers dict for the given URL.
    """
    return request(url, *args, **kwargs).headers

def sniff(url, *args, **kwargs):
    """ Returns the media type for the given URL.
    """
    return request(url, *args, **kwargs).headers.get('Content-Type', '').split(";")[0]

# print(sniff('http://www.textgain.com')) # 'text/html'

# Clear cache from 7+ days ago:

# t = time.time() - 7 * 24 * 60 * 60 
# for f in glob.glob(cd(CACHE, '*')):
#     if os.stat(f).st_mtime < t:
#         os.remove(f)

#---- SEARCH --------------------------------------------------------------------------------------
# The Google Search API grants 100 free requests per day.
# The Bing Search API grants 1,000 free requests per month.

# For unlimited (=paid) Google Search, Translate, Vision,
# you need to use your own API key with a payment method:
# https://console.developers.google.com/apis/dashboard

keys = {
    'Google' : 'AIzaSyCqy-sxq6gZGAKxsoSfDHvjHgaW-M__T94',
    'Bing'   : '78b5e99618mshdf023480e605b76p149c54jsn83432d326a65'
}

Result = collections.namedtuple('Result', ('url', 'text', 'date'))

def google(q, page=1, language='en', delay=1, cached=False, key=None):
    """ Returns an iterator of (url, description)-tuples from Google.
    """
    if 0 < page <= 10:
        r  = 'https://www.googleapis.com/customsearch/v1' # https://console.developers.google.com
        r += '?cx=000579440470800426354:6r3pb8h1s3m'      # https://cse.google.com
        r += '&key=%s' % (key or keys['Google'])
        r += '&q=%s'
        r += '&lr=lang_%s'
        r += '&start=%i'
        r += '&num=10'
        r %= (
            urllib.parse.quote(b(q)),
            urllib.parse.quote(b(language)), 1 + 10 * (page - 1))
        r = download(r, delay=delay, cached=cached)
        r = json.loads(u(r))

        for r in r.get('items', ()):
            yield Result(
                r['link'],
                r['snippet'],
                ''
            )

def bing(q, page=1, language='en', delay=1, cached=False, key=None):
    """ Returns an iterator of (url, description)-tuples from Bing.
    """
    if 0 < page <= 10:
        k = {
            'x-bingapis-sdk'  : 'true',
            'x-rapidapi-host' : 'bing-web-search1.p.rapidapi.com',
            'x-rapidapi-key'  : (key or keys['Bing']),
        }
        r  = 'https://bing-web-search1.p.rapidapi.com/search' # https://rapidapi.com
        r += '?q=%s'
        r += '&setLang=%s'
        r += '&offset=%s'
        r += '&count=10'
        r %= (
            urllib.parse.quote(b(q)),
            urllib.parse.quote(b(language)), 0 + 10 * (page - 1))
        r = download(r, headers=k, delay=delay, cached=cached)
        r = json.loads(u(r))

        for r in r.get('webPages', {}).get('value', ()):
            yield Result(
                r['url'],
                r['snippet'],
                r['dateLastCrawled']
            )

def search(q, engine='google', page=1, language='en', delay=1, cached=False, key=None):
    """ Returns an iterator of (url, description)-tuples.
    """
    f = globals().get(engine, lambda *args: None)
    if key:
        return f(q, page, language, delay, cached, key)
    else:
        return f(q, page, language, delay, cached)

# for r in search('textgain'):
#     print(r.url)
#     print(r.text)
#     print('\n')

#---- TRANSLATE -----------------------------------------------------------------------------------

def translate(q, source='', target='en', delay=1, cached=False, key=None):
    """ Returns the translated string.
    """
    r  = 'https://www.googleapis.com/language/translate/v2?'
    r += '&key=%s'    % (key or keys['Google'])
    r += '&q=%s'      % urllib.parse.quote(b(q[:1000]))
    r +=('&source=%s' % source) if source else ''
    r += '&target=%s' % target
    r  = download(r, delay=delay, cached=cached)
    r  = json.loads(u(r))
    r  = r.get('data', {})
    r  = r.get('translations', [{}])[0]
    r  = r.get('translatedText', '')
    return r

setattr(google, 'search'   , search   )
setattr(google, 'translate', translate)

# print(google.translate('De zwarte kat zat op de mat.', source='nl', key='***'))

#---- ANNOTATE ------------------------------------------------------------------------------------

Annotation = collections.namedtuple('Annotation', ('text', 'language', 'tags', 'location'))

def annotate(path, delay=1, key=None):
    """ Returns the image annotation.
    """
    with open(path, 'rb') as f:
        s = base64.b64encode(f.read())
        s = {'requests': [
                {'image': {'content': s}, 'features': [
                    {'type' :     'TEXT_DETECTION'},
                    {'type' :    'LABEL_DETECTION'},
                    {'type' : 'LANDMARK_DETECTION'},
                ]}
            ]}

    r  = 'https://vision.googleapis.com/v1/images:annotate?'
    r += 'key=%s' % (key or keys['Google'])
    r  = download(r, data=s, headers={'Content-Type': 'application/json'}, delay=delay)
    r  = json.loads(u(r))
    r  = r.get('responses'           , [{}])[0]
    r1 = r.get('textAnnotations'     , [{}])[0]
    r2 = r.get('labelAnnotations'    , [])      # lower
    r3 = r.get('landmarkAnnotations' , [])      # Title
    r4 = r.get('landmarkAnnotations' , [{}])[0] \
          .get('locations'           , [{}])[0] \
          .get('latLng')

    return Annotation(
        r1.get('description', '').strip(),
        r1.get('locale', ''),
        {w['description']: w['score'] for w in r2 + r3},
        (r4['latitude'], r4['longitude']) if r4 else None
    )

setattr(google, 'annotate', annotate)

# with tmp(download('http://goo.gl/5GTvTe'), 'wb') as f:
#     print(google.annotate(f.name, key='***'))

#---- TWITTER -------------------------------------------------------------------------------------

keys['Twitter'] = '78b5e99618mshdf023480e605b76p149c54jsn83432d326a65'

Tweet = collections.namedtuple('Tweet', ('id', 'text', 'date', 'language', 'author', 'likes'))

class Twitter(object):

    def search(self, q, language='', delay=4, cached=False, key=None, **kwargs):
        """ Returns an iterator of tweets.
        """
        k = {
            'X-RapidAPI-Host' : 'twitter-api45.p.rapidapi.com',
            'X-RapidAPI-Key'  : (key or keys['Twitter']),
        }
        r = 'https://twitter-api45.p.rapidapi.com/search.php' + \
            '?search_type=Latest' + \
            '&query='  + urllib.parse.quote(b(q)) + '%20lang:' + language + \
            '&cursor=' + ''

      # r = 'https://twitter-api45.p.rapidapi.com/timeline.php' + \
      #     '?screenname=' + q.lstrip('@')

        r = download(r, headers=k, delay=delay, cached=cached, **kwargs) # 1000/hr
        r = json.loads(r and u(r) or '{}')

        for v in r.get('timeline', ()):
            yield Tweet(
                v.get('tweet_id'   ) or '',
                v.get('text'       ) or '',
                v.get('created_at' ) or '',
              ( v.get('lang'       ) or '').replace('und', ''),
                v.get('screen_name') or '',
                v.get('favorites'  ) or '' 
            )

        # id = r.get('next_cursor')

    def __call__(self, *args, **kwargs):
        return self.search(*args, **kwargs)

twitter = Twitter()

# for tweet in twitter('cats', language='en'):
#     print(tweet.text)

#---- WIKIPEDIA -----------------------------------------------------------------------------------

# https://www.mediawiki.org/wiki/API:Parsing_wikitext

BIBLIOGRAPHY = {
    'style'              ,
    'div div'            , # figure
    'table'              , # infobox
    '#references'        , # references title
    '.reflist'           , # references
    '.reference'         , # [1]
    '.mw-editsection'    , # [edit]
    '.noprint'           , # [citation needed]
    '.mw-empty-elt'      ,
    'h2 < #see_also'     ,
    'h2 < #see_also ~ *' ,
    'h2 < #notes'        ,
    'h2 < #notes ~ *'    ,
}

def mediawiki(domain, q='', language='en', delay=1, cached=True, raw=False, **kwargs):
    """ Returns the HTML source of the given MediaWiki page (or '').
    """
    r  = 'https://%s.%s.org/w/api.php' % (language, domain)
    r += '?action=parse'
    r += '&format=json'
    r += '&redirects=1'
    r += '&prop=%s' % kwargs.get('prop', 'text')
    r += '&page=%s' % urllib.parse.quote(q)
    r  = download(r, delay=delay, cached=cached)
    r  = json.loads(u(r))

    if raw:
        return r['parse'] # for props
    else:
        return u'<h1>%s</h1>\n%s' % (
             u(r['parse']['title']),
             u(r['parse']['text']['*']))

def wikipedia(*args, **kwargs):
    """ Returns the HTML source of the given Wikipedia article.
    """
    return mediawiki('wikipedia', *args, **kwargs)

def wiktionary(*args, **kwargs):
    """ Returns the HTML source of the given Wiktionary article.
    """
    return mediawiki('wiktionary', *args, **kwargs)

# 1. Parse HTML source

# src = wikipedia('Arnold Schwarzenegger', language='en')
# dom = DOM(src) # see below

# 2. Parse article (full, plaintext):

# article = plaintext(dom, ignore=BIBLIOGRAPHY)
# print(article)

# 3. Parse bio (= 1st paragraph):

# bio = dom('p:not(.mw-empty-elt)')[0]
# bio = plaintext(bio)
# print(bio)

# 4. Parse links:

# for a in dom('a[href^="/wiki"]'):
#     a = a.href.split('/')[-1]
#     a = a.replace('_', ' ')
#     a = decode(a)
#     print(a)

# 5. Guess gender (he/she):

# s = ' %s ' % summary.lower()
# gender = s.count( ' he ') + \
#          s.count(' his ') > \
#          s.count(' she ') + \
#          s.count(' her ') and 'm' or 'f'
# print(gender)

# 6. Guess age:

# box = DOM(src)('.infobox')[0]
# age = plaintext(box('th:contains("born") + td span.ForceAgeToShow')[0])
# age = re.search(r'[0-9]+', age).group(0)
# age = int(age)
# print(age)

#---- GPT -----------------------------------------------------------------------------------------

# https://platform.openai.com/signup
# https://platform.openai.com/docs/guides/chat

keys['GPT'] = ''

class Q(unicode): pass
class A(unicode): pass

def gpt(q, d=1, delay=1, cached=False, timeout=30, key=None):
    """ Returns ChatGPT's response as a string.
    """
    if not isinstance(q, list): # [Q, A, ...] conversation
        q = [q]
    for i, s in enumerate(q):
        if type(s) in (Q, basestring):
            q[i] = { 'content': s, 'role': 'user'      }
        if type(s) in (A,):
            q[i] = { 'content': s, 'role': 'assistant' }

    r  = 'https://api.openai.com/v1/chat/completions', {
         'messages'      : q,
         'temperature'   : d,
         'model'         : 'gpt-4o-mini',
    }, { 'Content-Type'  : 'application/json',
         'Authorization' : 'Bearer %s' % (key or keys['GPT'])
    }
    r = download(*r, delay=delay, cached=cached, timeout=timeout)
    r = json.loads(u(r))
    n = r['usage']['total_tokens']
    r = r['choices'][0]
    r = r['message']
    r = r['content']
    r = r.strip()
  # print(n)
    return r

# q = 'You are a comedian. What is Earth?'
# q = 'Write like a child. What is Earth?'

# q = 'Write like a lawful evil dragon. Joke about capturing an adventurer.'

# print(gpt(q))

#---- GPS -----------------------------------------------------------------------------------------

keys['GPS'] = ''

Coordinates = collections.namedtuple('Coordinates', ('lat', 'lng'))

def gps(q, delay=35, cached=False, key=None):
    """ Returns the latitute & longitude of a location.
    """
    r  = 'https://api.opencagedata.com/geocode/v1/json'
    r += '?key=' + (key or keys['GPS'])
    r += '&q=' + urllib.parse.quote(b(q or '--'))
    r  = download(r, delay=delay, cached=cached) # 2500 / day
    r  = json.loads(u(r))
    r  = r.get('results')
    r  = iter(r)
    r  = next(r, {})
    r  = r.get('geometry', {})
    r  = r.get('lat'), \
         r.get('lng')
    r  = Coordinates(*r)
    r  = None if None in r else r
    return r

# print(gps('Aalst, BE'))
# print(gps('Aalst, NL'))

#---- RSS -----------------------------------------------------------------------------------------

Story = collections.namedtuple('Story', ('url', 'text', 'date', 'language', 'author'))

def rss(xml):
    """ Returns an iterator of stories from the given XML string (RSS feed).
    """
    t = ElementTree.fromstring(b(xml))
    for e in t.iter('item'):
        yield Story(
            u(e.findtext('link'             , '')),
            u(e.findtext('title'            , '')),
            u(e.findtext('pubDate'          , '')),
            u(t.findtext('channel/language' , '')).split('-')[0],
            u(e.findtext('author'           , 
              e.findtext('source'           , '')))
        )

def atom(xml, ns='http://www.w3.org/2005/Atom'):
    """ Returns an iterator of stories from the given XML string (Atom feed).
    """
    t = ElementTree.fromstring(b(xml))
    for e in t.iter('{%s}entry' % ns):
        yield Story(
            u(e.find    ('{%s}link'     % ns    ).get('href', '')),
            u(e.findtext('{%s}title'    % ns, '')),
            u(e.findtext('{%s}updated'  % ns, '')), u'',
            u(t.findtext('{%s}name'     % ns, ''))
        )

def feed(url, delay=1, cached=False, headers={'User-Agent': 'Grasp.py'}):
    s = download(url, headers=headers, delay=delay, cached=cached)
    for f in (rss, atom):
        try:
            for r in f(s):
                yield r
        except ElementTree.ParseError as e: # HTML?
            pass
        except:
            pass

# for story in feed('http://feeds.washingtonpost.com/rss/world'):
#     print(story)

#---- MAIL ----------------------------------------------------------------------------------------
# The mail() function sends a HTML-formatted e-mail from a Gmail account.

SMTP = collections.namedtuple('SMTP', ('username', 'password', 'server'))

def mail(to, subject, message, relay=SMTP('', '', 'smtp.gmail.com:465')):
    """ Sends a HTML e-mail using SSL encryption.
    """
    from email.mime.multipart import MIMEMultipart
    from email.mime.text      import MIMEText

    username, password, server = relay

    m = MIMEMultipart()
    m['From'] = username
    m['To'] = to
    m['Subject'] = subject
    m.attach(MIMEText(b(message), 'html', 'utf-8')) # html/plain
    m = m.as_string()

    s = smtplib.SMTP_SSL(server) # SSL
    s.login(username, password)
    s.sendmail(username, to, m)
    s.close()

# mail('grasp@mailinator.com', 'test', u'<b>Héllø</b>')

##### WWW #########################################################################################

#---- DOM -----------------------------------------------------------------------------------------
# The DOM or Document Object Model is a representation of a HTML document as a nested tree.
# The DOM can be searched for specific elements using CSS selectors.

# DOM('<div><p>hello</p></div>') results in:
# DOM([
#     Element('div', [
#         Element('p', [
#             Text('hello')
#         ])
#     ])
# ])

SELF_CLOSING = { # <img src="" />
    'area'    ,
    'base'    ,
    'br'      ,
    'col'     ,
    'command' ,
    'embed'   ,
    'hr'      ,
    'img'     ,
    'input'   ,
    'keygen'  ,
    'link'    ,
    'meta'    ,
    'param'   ,
    'source'  ,
    'track'   ,
    'wbr'     ,
}

def quote(s):
    """ Returns the quoted string.
    """
    if '"' in s:
        return "'%s'" % s
    else:
        return '"%s"' % s

class Node(object):

    def __init__(self):
        self.parent = None
        self.children = []

    def __iter__(self):
        return iter(self.children)

@printable
class Text(Node):

    def __init__(self, data):
        Node.__init__(self)
        self.data = data

    def __str__(self):
        return self.data

    def __repr__(self):
        return 'Text(%s)' % repr(self.data)

@printable
class Element(Node):

    def __init__(self, tag, attributes={}):
        Node.__init__(self)
        self.tag = tag
        self.attributes = collections.OrderedDict(attributes)

    def __getitem__(self, k):
        return self.attributes.get(k)  # a['href']

    def __getattr__(self, k):
        return self.attributes.get(k)  # a.href

    def __call__(self, css):
        return selector(self, css)     # div('a')

    def __repr__(self):
        return 'Element(tag=%s)' % repr(self.tag)

    def __str__(self):
        a = ' '.join('%s=%s' % (k, quote(v)) for k, v in self.attributes.items() if v is not None)
        a = ' ' + a if a else ''
        if self.tag in SELF_CLOSING:
            return u'<%s%s />' % (
                self.tag, a)
        else:
            return u'<%s%s>%s</%s>' % (
                self.tag, a, self.html, self.tag)

    @property
    def html(self):
        return ''.join(u(n) for n in self)

    @property
    def successors(self):
        if self.parent:
            for n in self.parent.children[self.parent.children.index(self) + 1:]:
                if isinstance(n, Element):
                    yield n

    @property
    def predecessors(self):
        if self.parent:
            for n in self.parent.children[:self.parent.children.index(self)]:
                if isinstance(n, Element):
                    yield n

    @property
    def next(self):
        """ Yields the next sibling in Element.parent.children.
        """
        return next(self.successors, None)

    @property
    def previous(self):
        """ Yields the previous sibling in Element.parent.children.
        """
        return next(self.predecessors, None)

    def match(self, tag='*', attributes={}):
        """ Returns True if the element has the given tag and attributes.
        """
        if tag != '*':
            if tag != self.tag:
                return False
        for k, v in attributes.items():
            if self[k] is None:
                return False
            if self[k] != v and not type(v) is REGEX:
                return False
            if self[k] != v and not v.search(self[k]):
                return False
        return True

    def find(self, tag='*', attributes={}, depth=10*10):
        """ Returns an iterator of nested elements with the given tag and attributes.
        """
        if depth > 0:
            for n in self:
                if isinstance(n, Element):
                    if n.match(tag, attributes):
                        yield n
                    for n in n.find(tag, attributes, depth-1):
                        yield n

@printable
class Document(HTMLParser, Element):

    def __init__(self, html):
        """ Document Object Model, a tree of Element and Text nodes from the given HTML string.
        """
        HTMLParser.__init__(self)
        Element.__init__(self, tag=None)
        self.head = None
        self.body = None
        self.type = None
        self._stack = [self]
        self.feed(u(html))

    def __repr__(self):
        return 'Document()'

    def __str__(self):
        return (self.type or '') + self.html

    def handle_decl(self, decl):
        self.type = '<!%s>' % decl

    def handle_entityref(self, name):
        self.handle_data('&%s;' % name)

    def handle_charref(self, name):
        self.handle_data('&#%s;' % name)

    def handle_data(self, data):
        try:
            n = Text(data)
            n.parent = self._stack[-1]
            n.parent.children.append(n)
        except:
            pass

    def handle_starttag(self, tag, attributes):
        try:
            n = Element(tag, attributes)
            n.parent = self._stack[-1]
            n.parent.children.append(n)
            # New elements will be nested inside,
            # unless it is self-closing (<br />).
            if tag not in SELF_CLOSING:
                self._stack.append(n)
        except:
            pass

    def handle_endtag(self, tag):
        try:
            if tag not in SELF_CLOSING:
                n = self._stack.pop()
            if n.tag == 'head':
                self.head = n
            if n.tag == 'body':
                self.body = n
        except:
            pass

DOM = Document
dom = Document

# dom = DOM(download('https://www.textgain.com'))
# 
# for a in dom.find('a'):
#     print(1, a.href)
#
# for a in dom.find('a', {'href': re.compile(r'^https://')}):
#     print(2, a.href)

#---- CSS SELECTORS -------------------------------------------------------------------------------
# CSS selectors (http://www.w3schools.com/cssref/css_selectors.asp) yield a list of child elements.
# For example div('a.external') returns a list of <a class="external"> elements in the given <div>.

# The approach is very powerful to build HTML crawlers & parsers.
# Here is the age of Arnold Schwarzenegger parsed from Wikipedia:

# s = download('https://en.wikipedia.org/wiki/Arnold_Schwarzenegger')
# t = DOM(s)
# t = t('table.infobox')[0]
# t = t('th:contains("born") + td')[0]  # <th>Born:<th><td> ... </td>
# s = plaintext(t)
# print(s)

SELECTOR = re.compile(''.join((
    r'^',
    r'([+<>~])?',                                                 # combinator + < >
    r'(\w+|\*)?',                                                 # tag
    r'((?:[.#][-\w]+)|(?:\[.*?\]))?',                             # attributes # . [=]
    r'(\:first-child|\:(?:nth-child|not|has|contains)\(.*?\))?',  # pseudo :
    r'$'
)))

CLASS = \
    r'(^|\s)%s(\s|$)'

def selector(element, s):
    """ Returns a list of nested elements that match the given CSS selector chain.
    """
    m = []
    s = s.strip()
    s = s.lower()                                                 # case-insensitive
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'([,+<>~])\s', '\\1', s)
    s = s or '<>'

    for s in s.split(','):                                        # div, a
        e = [element]
        for s in re.split(r' (?![^\[\(]*[\]\)])', s):             # div a[class="x y"]

            try:
                combinator, tag, a, pseudo = \
                    SELECTOR.search(s).groups('')
            except: 
                return []
            tag = tag or '*'                                      # *

            if not a:                                             # a
                a = {}
            elif a.startswith('#'):                               # a#id
                a = {   'id': re.compile(        a[1:], re.I)}
            elif a.startswith('.'):                               # a.class
                a = {'class': re.compile(CLASS % a[1:], re.I)}
            elif a.startswith('['):                               # a[href]
                a = a.strip('[]')
                a = a.replace('"', '')
                a = a.replace("'", '')

                k, op, v = (re.split(r'([\^\$\*]?=)', a, 1) + ['=', r'.*'])[:3]

                if op ==  '=':
                    a = {k: re.compile(r'^%s$' % v, re.I)}        # a[href="https://textgain.com"]
                if op == '^=':
                    a = {k: re.compile(r'^%s'  % v, re.I)}        # a[href^="https"]
                if op == '$=':
                    a = {k: re.compile(r'%s$'  % v, re.I)}        # a[href$=".com"]
                if op == '*=':
                    a = {k: re.compile(r'%s'   % v, re.I)}        # a[href*="textgain"]

            if combinator == '':
                e = (e.find(tag, a) for e in e)
                e = list(itertools.chain(*e))                     # div a
            if combinator == '>':
                e = (e.find(tag, a, 1) for e in e)
                e = list(itertools.chain(*e))                     # div > a
            if combinator == '<':
                e = [e for e in e if any(e.find(tag, a, 1))]      # div < a
            if combinator == '+':
                e = map(lambda e: e.next, e)
                e = [e for e in e if e and e.match(tag, a)]       # div + a
            if combinator == '~':
                e = map(lambda e: e.successors, e)
                e = (e for e in e for e in e if e.match(tag, a))
                e = list(unique(e))                               # div ~ a

            if pseudo.startswith(':first-child'):
                e = (e for e in e if not e.previous)
                e = list(unique(e))                               # div a:first-child
            if pseudo.startswith(':nth-child'):
                s = pseudo[11:-1]
                e = [e[int(s) - 1]]                               # div a:nth-child(2)
            if pseudo.startswith(':not'):
                s = pseudo[5:-1]
                e = [e for e in e if e not in element(s)]         # div:not(.main)
            if pseudo.startswith(':has'):
                s = pseudo[5:-1]
                e = [e for e in e if e(s)]                        # div:has(.main)
            if pseudo.startswith(':contains'):
                s = pseudo[11:-2]
                e = (e for e in e if s in e.html.lower())
                e = list(unique(e))                               # div:contains("hello")

        m.extend(e)
    return m

# dom = DOM(download('https://www.textgain.com'))
# 
# print(dom('nav a')[0].html)
# print(dom('a[href^="https"]:first-child'))
# print(dom('a:contains("open source")'))

#---- PLAINTEXT -----------------------------------------------------------------------------------
# The plaintext() function traverses a DOM HTML element, strips all tags while keeping Text data.

BLOCK = {
    'article'    ,
    'aside'      ,
    'blockquote' ,
    'center'     ,
    'div'        ,
    'dl'         ,
    'figure'     ,
    'figcaption' ,
    'footer'     ,
    'form'       ,
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'header'     , 
    'hr'         ,
    'main'       ,
    'ol'         ,
    'p'          ,
    'pre'        ,
    'section'    ,
    'title'      ,
    'table'      ,
    'textarea'   ,
    'ul'         ,
}

PLAIN = {
     'li' : lambda s: '* %s\n' % re.sub(r'\n\s+', '\n&nbsp;&nbsp;', s),
     'h1' : lambda s: s + '\n' + "-" * len(s),
     'h2' : lambda s: s + '\n' + "-" * len(s),
     'br' : lambda s: s + '\n'  ,
     'tr' : lambda s: s + '\n\n',
     'th' : lambda s: s + '\n'  ,
     'td' : lambda s: s + '\n'  ,
}

def plaintext(element, keep={}, ignore={'head', 'script', 'style', 'form'}, format=PLAIN):
    """ Returns the given element as a plaintext string.
        A (tag, [attributes])-dict to keep can be given.
    """
    if not isinstance(element, Element): # str?
        element = DOM(element)

    # CSS selectors in ignore list (e.g., form.submit)
    ignore = set(selector(element, ', '.join(ignore)))

    def r(n): # node recursion
        s = ''
        for n in n:
            if isinstance(n, Text):
                # Collapse spaces, decode entities (&amp;)
                ch = re.sub(r'\s+', ' ', unescape(n.data))
            if isinstance(n, Element):
                if n not in ignore:
                    ch = r(n)
                else:
                    continue
                if n.tag in BLOCK:
                    ch = ch.strip()
                if n.tag in keep:
                    a  = ' '.join('%s=%s' % (k, quote(n[k])) for k in keep[n.tag] if n[k] != None)
                    a  = ' ' + a if a else ''
                    ch = '<%s%s>%s</%s>' % (n.tag, a, ch, n.tag)
                else:
                    ch = format.get(n.tag, lambda s: s)(ch)
                if n.tag in BLOCK:
                    ch = '\n\n%s\n\n' % ch
            s += ch
        return s

    s = r(element)
    s = re.sub(r'(\s) +'       , '\\1'  , s) # no left indent
    s = re.sub(r'&nbsp;'       , ' '    , s) # exdent bullets
    s = re.sub(r'\n +\*'       , '\n*'  , s) # dedent bullets
    s = re.sub(r'\n\* ?(?=\n)' , ''     , s) # no empty lists
    s = re.sub(r'\n\n+'        , '\n\n' , s) # no empty lines
    s = s.strip()
    return s

plain = plaintext

# dom = DOM(download('https://www.textgain.com'))
# txt = plaintext(dom, keep={'a': ['href']})
# 
# print(txt)

def encode(s):
    """ Returns a string with encoded entities.
    """
    s = s.replace('&' , '&amp;'  )
    s = s.replace('"' , '&quot;' )
    s = s.replace("'" , '&apos;' )
    s = s.replace('<' , '&lt;'   )
    s = s.replace('>' , '&gt;'   )
  # s = s.replace('\n', '&#10;'  )
  # s = s.replace('\r', '&#13;'  )
    return s

def decode(s):
    """ Returns a string with decoded entities.
    """
    s = s.replace('&nbsp;' , ' ' )
    s = s.replace('&quot;' , '"' )
    s = s.replace('&apos;' , "'" )
    s = s.replace('&#39;'  , "'" )
    s = s.replace('&lt;'   , '<' )
    s = s.replace('&gt;'   , '>' )
    s = s.replace('&amp;'  , '&' )
  # s = s.replace('&#10;'  , '\n')
  # s = s.replace('&#13;'  , '\r')
    s = re.sub(r'https?://.*?(?=\s|$)', \
        lambda m: urllib.parse.unquote(m.group()), s) # '%3A' => ':' (in URL)
    return s

def docx(path):
    """ Returns the given .docx file as a plain text string.
    """
    x = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    f = zipfile.ZipFile(path)
    f = f.open('word/document.xml')
    t = ElementTree.parse(f)
    s = ''
    for e in t.iter(x + 'p'):
        for e in e.iter(x + 't'):
            s += e.text
        s = s.rstrip()
        s = s + '\n\n'
    return s

def pdf(path, encoding='UTF-8'):
    """ Returns the given .pdf file as a plain text string.
    """
    if OS == 'Linux':
        p = next(ls('pdftotext-linux'  ), '') # Xpdf tools 
    if OS == 'Darwin':
        p = next(ls('pdftotext-mac'    ), '')
    if OS == 'Windows':
        p = next(ls('pdftotext-win.exe'), '')
    if not p:
        p = next(ls('pdftotext'))

    s = subprocess.check_output((p, '-q', '-enc', encoding, path, '-'))
    s = u(s)
    return s

#---- NEWS ----------------------------------------------------------------------------------------
# The customary schema for articles is <div itemprop="articleBody">, but there are many exceptions.
# The article() function returns tidy plaintext for most articles from known newspapers.

ARTICLE = (
# CSS SELECTOR                                   USED BY:
 'article[class*="article"]'                 , # The Sun
 'article[itemprop="articleBody"]'           , # United Press International (UPI)
    'span[itemprop="articleBody"]'           , # La Repubblica
     'div[itemprop="articleBody"]'           , # Le Monde
     'div[id="rcs-articleContent"] .column1' , # Reuters
     'div[id*="storyBody"]'                  , # Academic Press (AP)
     'div[id*="article_body"]'               , # Gazeta Wyborcza
     'div[is$="article-body"]'               , # The Onion
     'div[class*="story-body"]'              , # New York Times
     'div[class*="entry-text"]'              , # Huffington Post
     'div[class*="text"] article'            , # Yomiuri Shimbun
     'div[class*="article-body"]'            , # Le Soir
     'div[class^="article-section"]'         , # Der Spiegel
     'div[class^="article_"]'                , # De Standaard
     'div[class^="article-"]'                , # Aftonbladet
     'div[class*="article"]'                 , # Dainik Bashkar
     'article'                               , # Bild
     '.news-detail'                          , # Hurriyet
     '.postBody'                             ,
     '.story'                                ,
)

SOCIAL = (
  'script'                                   ,
   'style'                                   ,
    'form'                                   ,
   'aside[class*="share"]'                   ,
     'div[class*="share"]'                   ,
      'ul[class*="share"]'                   ,
      'li[class*="share"]'                   ,
     'div[class*="social"]'                  ,
      'ul[class*="social"]'                  ,
       '*[class*="pagination"]'              ,
       '*[class*="gallery"]'                 ,
       '*[class*="photo"]'                   ,
       '*[class*="video"]'                   ,
       '*[class*="button"]'                  ,
       '*[class*="signup"]'                  ,
       '*[class*="footer"]'                  ,
       '*[class*="hidden"]:not(.field-label-hidden)', # <div class="visually-hidden"> (NYT)
       '*[class*="module"]'                  ,
       '*[class*="widget"]'                  ,
       '*[class*="meta"]'                    ,
)

def article(url, cached=False, headers={'User-Agent': 'Grasp.py'}):
    """ Returns a (title, article)-tuple from the given online newspaper.
    """
    s  = download(url, cached=cached, headers=headers)
    t  = DOM(s)
    e1 = t('article h1, h1:not(.logo), h1, h2, .entry-title')
    e2 = t(', '.join(ARTICLE))
    e1 = next(iter(e1), '')
    e2 = next(iter(e2), '')
    s1 = plaintext(e1).strip('"\'')    # article title
    s2 = plaintext(e2, ignore=SOCIAL)  # article text
    s2 = re.sub('\n--+\n\n', '\n\n', s2)
    return s1, s2

# url = 'http://rss.nytimes.com/services/xml/rss/nyt/World.xml'
# for story in feed(url):
#     title, text = article(story.url, cached=True)
#     print(title.upper() + '\n')
#     print(text + '\n\n')

###################################################################################################

#---- DATE ----------------------------------------------------------------------------------------
# The date() function attempts to parse a Date object from a string or timestamp (int/float).

DATE = (
#    http://strftime.org           # DATE                            USED BY:
    '%a %b %d %H:%M:%S +0000 %Y' , # Mon Jan 31 10:00:00 +0000 2000  Twitter
    '%Y-%m-%dT%H:%M:%S+0000'     , # 2000-01-31T10:00:00+0000        Facebook
    '%Y-%m-%dT%H:%M:%SZ'         , # 2000-01-31T10:00:00Z            Bing
    '%Y-%m-%d %H:%M:%S'          , # 2000-01-31 10:00:00
    '%Y-%m-%d %H:%M'             , # 2000-01-31 10:00
    '%Y-%m-%d'                   , # 2000-01-31
)

def rfc_2822(s):                   # Mon, 31 Jan 2000 10:00:00 GMT   RSS
    return email.utils.mktime_tz(
           email.utils.parsedate_tz(s))

class DateError(Exception):
    pass

@printable
class Date(datetime.datetime):

    # Date.year
    # Date.month
    # Date.day
    # Date.hour
    # Date.minute
    # Date.second

    @property
    def week(self):
        return self.isocalendar()[1]

    @property
    def weekday(self):
        return self.isocalendar()[2] # Mon = 1

    @property
    def timestamp(self):
        return time.mktime(self.timetuple()) + self.microsecond * 0.000001

    def format(self, s):
        return self.strftime(s)

    def __str__(self):
        return self.strftime('%Y-%m-%d %H:%M:%S')

    def __int__(self):
        return self.timestamp.__int__()

    def __float__(self):
        return self.timestamp

    def __add__(self, t):
        """ Returns a new date t seconds after.
        """
        return date(datetime.datetime.__add__(self, 
                    datetime.timedelta(seconds=t) if type(t) in (int, float) else t))

    def __sub__(self, t):
        """ Returns a new date t seconds before.
        """
        return date(datetime.datetime.__sub__(self, 
                    datetime.timedelta(seconds=t) if type(t) in (int, float) else t))

    def __repr__(self):
        return "Date(%s)" % repr(self.__str__())

def date(*v, **format):
    """ Returns a Date from the given timestamp or date string.
    """
    format = format.get('format', '%Y-%m-%d %H:%M:%S')

    if len(v) >  1:
        return Date(*v) # (year, month, day, ...)
    if len(v) <  1:
        return Date.now()
    if len(v) == 1:
        v = v[0]
    if isinstance(v, datetime.datetime) and PY3:
        return Date.combine(v.date(), v.time(), v.tzinfo)
    if isinstance(v, datetime.datetime) and PY2:
        return Date.combine(v.date(), v.time())
    if isinstance(v, int):
        return Date.fromtimestamp(v)
    if isinstance(v, float):
        return Date.fromtimestamp(v)
    if isinstance(v, tuple):
        return Date(*v)
    if v is None:
        return Date.now()
    try:
        return Date.fromtimestamp(rfc_2822(v)) 
    except: 
        pass
    for f in (format,) + DATE:
        try:
            return Date.strptime(v, f)
        except:
            pass
    raise DateError('unknown date format: %s' % repr(v))

def datediff(d1, d2):
    """ Returns a timedelta for the difference between d1 - d2.
    """
    return datetime.datetime.__sub__(d1, d2)

def modified(path):
    """ Returns a Date for the given file path (last modified).
    """
    if os.path.exists(path):
        return date(os.path.getmtime(path))

def tz(d):
    """ Returns a Date with tzinfo (timezone-aware) if missing.
    """
    if not d.tzinfo:
        return date(d.replace(tzinfo=datetime.timezone.utc))
    else:
        return date(d)

def timeline(a, n=10, date=lambda v: v, since=None, until=None):
    """ Returns an ordered dict of n lists distributed by date.
    """
    m = collections.OrderedDict()
    n = round(n)
    a = list(a)
    t = list(map(date, a))
    a = zip(t, a)
    x = min(t)
    y = max(t)
    x = float(since or x)
    y = float(until or y)
    d = float(y - x) / (n - 1 or 1) # interval
    # Binning:
    if n <= 0:
        a = []
    if n == 1:
        d = x
    for i in range(n):
        t = x + i * d
        t = Date.fromtimestamp(t)
        m[t] = []
    for t, v in a:
        t = x + mround(float(t) - x, d)
        t = Date.fromtimestamp(t)
        m[t].append(v)
    return m

tl = timeline

# a = (
#     date(2023,  1, 1),
#     date(2023,  1, 2),
#     date(2023,  1, 3),
# )
# for k, v in tl(a, n=2.6).items():
#     print(k, len(v))

#---- DATE IN TEXT --------------------------------------------------------------------------------

temporal = {
    'en': ((
        r'(after|before|during)'                               ,
        r'(at|in|of|on|to|from)'                               ,
        r'(the|this|a|an)'                                     ,
        r'(one|two|three|four|five|six|seven|eight|nine|ten)'  ,
        r'(first|last|past|next|previous|coming)'              ,
        r'\d{1,4}(st|nd|rd|th)?'                               ,
        r'\d+-\d+'                                             ,
    ), (
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'   ,
        r'(January|February|March|April|May|June|July|August)' ,
        r'(Septem|Octo|Novem|Decem)ber'                        ,
        r'(spring|summer|autumn|winter)'                       ,
        r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun)'                       ,
        r'(Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)days?'          ,
        r'(minute|hour|day|week|weekend|month|year)s?|century' ,
        r'(morning|midday|afternoon|evening|midnight|night)'   ,
        r'(yesterday|today|tonight|tomorrow)'                  ,
        r'\d+ ?(yr?|hr|min|sec)s?'                             , # 1hr
        r'\d\d?(am|pm)'                                        , # 1am
        r'\d\d?:\d\d(am|pm)?'                                  , # dd:mm
        r'\d\d?[/-]\d\d?[/-]\d{4}'                             , # dd/mm/yyyy
        r'\d{4}[/-]\d\d?[/-]\d\d?'                             , # yyyy/mm/dd
        r'\d{4}[-–]\d{4}'                                      , # yyyy-yyyy
        r'(19|20)\d\d'                                         , # yyyy
        r'(AD|BC)'                                             ,
    ), (
        r'(am|pm|a\.m\.?|p\.m\.?|o\'?clock|CEST|CET|GMT|UTC)'  ,
        r'(one|two|three|four|five|six|seven|eight|nine|ten)'  ,
        r'(ago|after|before)'                                  ,
        r'\d{1,4}(st|nd|rd|th)?'                               ,
    ))
}

_RE_TEMP = r'((%s )*(%s )+(%s )?)+'

def when(s, language='en'):
    """ Returns a list of temporal references in the string.
    """
    s = s + ' '
    s = re.sub(r'([,.!?)])', ' \\1', s)
    s = s.replace('a .m .', 'a.m.')
    s = s.replace('p .m .', 'p.m.') 
    r = temporal.get(language, ('   '))
    r = ('|'.join(r) for r in r)
    r = ('(%s)' % r  for r in r)
    r = _RE_TEMP % tuple(r)
    m = re.finditer(r, s, re.I)
    m = (m.group() for m in m)
    m = (m.strip() for m in m)
    m = (m for m in m if m)
    m = list(m)
    return m

# print(when('the day after tomorrow'))

##### WWW #########################################################################################

#---- APP -----------------------------------------------------------------------------------------
# The App class can be used to create a web service or GUI, served in a browser using JSON or HTML.

SECOND, MINUTE, HOUR, DAY = 1, 1*60, 1*60*60, 1*60*60*24

STATUS = BaseHTTPServer.BaseHTTPRequestHandler.responses
STATUS = {int(k): v[0] for k, v in STATUS.items()}
STATUS[429] = 'Too Many Requests'

headers = {
    'Content-Type': 
        'text/html; charset=utf-8',
    'X-Frame-Options': 
        'sameorigin',
    'Access-Control-Allow-Origin': # CORS
        '*',
    'Access-Control-Allow-Headers':
        '*',
}

# Recycle threads for handling concurrent requests:
class ThreadPoolMixIn(SocketServer.ThreadingMixIn):

    def __init__(self, threads=10):
        self.pool = multiprocessing.pool.ThreadPool(threads)

    def process_request(self, *args):
        self.pool.apply_async(self.process_request_thread, args)

    def process_request_thread(self, *args):
        try:
            SocketServer.ThreadingMixIn.process_request_thread(self, *args)
        except socket.timeout:
            pass
        except socket.error:
            pass

    def handle_error(self, *args):
      # traceback.print_exc()
        pass

class RouteError(Exception):
    pass

class Router(dict):

    def __init__(self, static=None):
        self.static = static

    def __setitem__(self, path, f):
        """ Defines the handler function for the given path.
        """
        return dict.__setitem__(self, path.strip('/'), f)

    def __getitem__(self, path):
        """ Returns the handler function for the given path.
        """
        return dict.__getitem__(self, path.strip('/'))

    def __call__(self, path, query):
        """ Returns the handler's value for the given path,
            or for the parent path if no handler is found.
        """
        path = path.strip('/')
        path = path.split('/')

        # static:
        if self.static:
            b = os.path.join(self.static, '') # add /
            f = os.path.join(self.static, *path)
            b = os.path.realpath(b)
            f = os.path.realpath(f)
            if f.startswith(b) and os.path.isfile(f):
                return open(f, 'rb')

        # dynamic:
        for i in reversed(range(len(path) + 1)):
            f = self.get('/'.join(path[:i]))
            if f:
                return f(*path[i:], **query)

        raise RouteError

class HTTPRequest(threading.local):

    def __init__(self):
        self.app     = None
        self.ip      = None
        self.method  = 'GET'
        self.path    = '/'
        self.query   = {}
        self.headers = {}

class HTTPResponse(threading.local):

    def __init__(self):
        self.code    = 200
        self.headers = {}

class HTTPError(Exception):

    def __init__(self, code=404):
        self.code = code

class FormData(dict):

    def __init__(self, s):
        self.update(re.findall(
            r'\r\nContent-Disposition: form-data; name="(.*?)"\r\n\r\n(.*?)\r\n', u(s)))

def generic(code, traceback=''):
    return '<h1>%s %s</h1><pre>%s</pre>' % (code, STATUS[code], traceback)

def mimetype(url):
    return mimetypes.guess_type(url)[0] or 'application/octet-stream'

WSGIServer, WSGIRequestHandler = (
    wsgiref.simple_server.WSGIServer,
    wsgiref.simple_server.WSGIRequestHandler
)

class App(ThreadPoolMixIn, WSGIServer):

    def __init__(self, host='127.0.0.1', port=8080, root=None, session=HOUR, threads=10, debug=False):
        """ A multi-threaded web app served by a WSGI-server, that starts with App.run().
        """
        ThreadPoolMixIn.__init__(self, threads)

        WSGIServer.__init__(self, (host, port), WSGIRequestHandler, bind_and_activate=False)

        self.set_app(self.__call__)
        self.rate     = collections.defaultdict(lambda: (0, 0)) # (used, since)
        self.router   = Router(static=root)
        self.request  = HTTPRequest()
        self.response = HTTPResponse()
        self.state    = HTTPState(self, expires=session)
        self.generic  = generic
        self.debug    = debug
        self.up       = False # running?

    def run(self, host=None, port=None, debug=True):
        """ Starts the server.
        """
        if host is None:
            host = self.server_address[0]
        if port is None:
            port = self.server_address[1]
        if debug:
            autoreload(self)

        print('Starting server at %s:%s... press ctrl-c to stop.' % (host, port))

        self.up = True
        self.debug = debug
        self.server_address = host, port
        self.server_bind()
        self.server_activate()
        self.serve_forever()

    def route(self, path, rate=None, key=lambda request: request.ip):
        """ The @app.route(path) decorator defines the handler for the given path.
            The handler(*path, **query) returns a str or dict for the given path. 
            With rate=(n, t), the IP address is granted n requests per t seconds, 
            before raising a 429 Too Many Requests error. A dict with {key: n, t} 
            individual rates can also be given.
        """
        def decorator(f):
            def wrapper(*args, **kwargs):
                if rate is not None:
                    k = key(self.request)       # user id
                    v = rate # (nonlocal)
                    try:
                        v = v(k) if callable(v) else v
                    except:
                        raise HTTPError(403)
                    try:
                        v = v[k] if isinstance(v, dict) else v
                    except KeyError:
                        raise HTTPError(403)
                    try:
                        n, t = self.rate[k]     # used, since (~0.1KB)
                    except KeyError:
                        raise HTTPError(403)
                    if v[1] <= time.time() - t: # now - since > interval?
                        t, n = time.time() , 0  # used = 0
                    if v[0] <= n:               # used > limit?
                        raise HTTPError(429)
                    self.rate[k] = n + 1, t     # used + 1
                return f(*args, **kwargs)
            self.router[path] = wrapper
            return wrapper
        return decorator

    def error(self, f):
        self.generic = f

    def __call__(self, env, start_response):

        self.up = True

        # Parse HTTP headers.
        # 'HTTP_USER_AGENT' => 'User-Agent'
        def parse(env):
            for k, v in env.items():
                if k[:5] == 'HTTP_':
                    k = k[5:]
                    k = k.replace('_', '-')
                    k = k.title()
                    yield u(k), u(v)

        # Parse HTTP GET and POST data.
        # '?page=1' => (('page', '1'),)
        def query(env):
            h1 = env.get('CONTENT_TYPE'  )
            h2 = env.get('CONTENT_LENGTH')
            q1 = env.get('QUERY_STRING'  ) # GET 
            q2 = env.get('wsgi.input'    ) # POST

            if h1.startswith('multipart') is False:
                q2 = q2.read(int(h2 or 0))
            if h1.startswith('multipart') is True:
                q2 = q2.read(int(h2 or 0))
                q2 = FormData(q2)
            if h1.startswith('application/json'):
                q2 = json.loads(q2 or '{}')
            if hasattr(q2, 'keys'):
                q2 = {k: [q2[k]] for k in q2}
            if isinstance(q1, bytes):
                q1 = u(q1)
            if isinstance(q2, bytes):
                q2 = u(q2)
            if isinstance(q1, unicode):
                q1 = urllib.parse.parse_qs(q1, True)
            if isinstance(q2, unicode):
                q2 = urllib.parse.parse_qs(q2, True)
            for k, v in q1.items():
                yield k, v[-1]
            for k, v in q2.items():
                yield k, v[-1]

        # Set App.request (thread-safe).
        r = self.request
        r.__dict__.clear()
        r.__dict__.update({
            'app'     : self,
            'env'     : env,
            'ip'      : env['REMOTE_ADDR'],
            'method'  : env['REQUEST_METHOD'],
            'path'    : env['PATH_INFO'].encode('iso-8859-1').decode(),
            'query'   : dict(query(env)), # bugs.python.org/issue16679 
            'headers' : dict(parse(env))
        })

        # Set App.response (thread-safe).
        r = self.response
        r.__dict__.clear()
        r.__dict__.update({
            'headers' : headers.copy(),
            'code'    : 200
        })

        try:
            v = self.router(self.request.path, self.request.query)
        except Exception as e:
            if isinstance(e, HTTPError):
                r.code = e.code
            elif isinstance(e, RouteError):
                r.code = 404
            elif isinstance(e, NotFound):
                r.code = 404
            elif isinstance(e, Forbidden):
                r.code = 403
            elif isinstance(e, TooManyRequests):
                r.code = 429
            else:
                r.code = 500
            v = self.generic(r.code, traceback.format_exc() if self.debug else '')

        if isinstance(v, (dict, list)):
            r.headers['Content-Type' ] = 'application/json; charset=utf-8'
            v = json.dumps(v)
        if hasattr(v, 'name') and \
           hasattr(v, 'read'):
            r.headers['Content-Type' ] = mimetype(v.name)
            r.headers['Last-Modified'] = modified(v.name).format('%a, %d %b %Y %H:%M:%S GMT')
            v = v.read()
        if v is None:
            v = ''

        # https://www.python.org/dev/peps/pep-0333/#the-start-response-callable
        start_response('%s %s' % (r.code, STATUS[r.code]), list(r.headers.items()))
        return [b(v)]

# app = application = App(threads=10)

# http://127.0.0.1:8080/products?page=1
# @app.route('/')
# def index(*path, **query):
#     #raise HTTPError(500)
#     #db = Database('data.db') # fast & safe for reading
#     return 'Hello world! %s %s' % (repr(path), repr(query))

# http://127.0.0.1:8080/api/tag?q=Hello+world!&language=en
# @app.route('/api/tag', rate=(10, MINUTE))
# def api_tag(q='', language='en'):
#     return tag(q, language)

# @app.error
# def error(code, traceback=''):
#     return 'Oops!'

# sys.stderr = os.devnull # (silent)

# app.run()

#---- APP RELOAD ----------------------------------------------------------------------------------

def autoreload(app):
    """ Restarts the script running the app when modified.
    """
    f = [ inspect.getouterframes(
          inspect.currentframe())[2][1], date() ] # caller

    @scheduled(1)
    def _():
        try:
            t = modified(f[0])
            if t < f[1]:
                return
            if sys.platform.startswith('win'):
                return
            f[1] = t
            app.shutdown()
            app.server_close() 
            os.execl(
                sys.executable,
                sys.executable,
               *sys.argv)
        except:
            pass

#---- APP SESSION ---------------------------------------------------------------------------------

class Cookie(dict):

    def __init__(self, s='', **kwargs):
        """ Returns the cookie string as a dict.
        """
        def parse(s):
            for s in s.split(';'):
                s = s.split('=') 
                s = s + ['']
                k = s[0].strip()
                v = s[1].strip()
                if k:
                    yield k, v
        if isinstance(s, basestring):
            s = parse(s)
        self.update(s)
        self.update(kwargs)

    def __str__(self):
        return ';'.join('%s=%s' % (k, v) if v != '' else k for k, v in self.items())

class HTTPState(dict):

    def __init__(self, app, expires=HOUR):
        """ HTTP state management, through cookies.
        """
        self.app     = app
        self.expires = expires
        self.checked = 0

    def __call__(self):
        """ Returns a dict for the current session.
        """
        t = time.time()
        if -self.checked + t > self.expires * 0.1:
            self.checked = t
            self.expire(t)
        try:
            k = self.app.request.headers['Cookie']
            k = Cookie(k)
            k = k['session'] # id
        except:
            k = binascii.hexlify(os.urandom(32))
            k = u(k)

        # Update session expiry:
        self[k] = v = t, self.get(k, [{}])[-1]

        # Update session cookie:
        self.app.response.headers['Set-Cookie'] = ''.join((
            'session=%s;' % k,
            'Max-Age=%s;' % self.expires,
            'SameSite=Strict;',
            'Secure;'
        ))
        return v[1]

    def expire(self, t=0.0):
        for k in list(self):
            if self.get(k, [t])[0] + self.expires <= t: # inactive?
                self.pop(k, None)

# App.request.session loads App.state() lazily.
# App.request.session is local to every thread.
# App.session hot links to App.request.session:

def _lazy_state(r):
    try:
        v = r._state
    except:
        v = r._state = r.app.state() \
                    if r.app else {}
    return v

HTTPRequest.session = \
    property(_lazy_state)

App.session = \
    property(lambda self: self.request.session)

# app = App()
# 
# @app.route('/')
# def index(*path, **query):
#     p = app.session
#     print(p)
#     if 'pw' not in p:
#         p['pw'] = 123
# 
# app.run()

#---- APP DATA ------------------------------------------------------------------------------------

def dataurl(path):
    """ Returns a data URL from the given file path.
    """
    k = mimetype(path) # 'image/png'
    v = open(path, 'rb')
    v = v.read()
    v = base64.b64encode(v)
    v = v.decode('utf-8')
    v = 'data:%s;base64,%s' % (k, v)
    return v

##### NET #########################################################################################

#---- GRAPH ---------------------------------------------------------------------------------------
# A graph is a set of nodes and edges (i.e., connections between nodes).
# A graph is stored as an adjacency matrix {node1: {node2: edge weight}}.
# It can then be analysed to find the shortest paths (e.g., navigation),
# clusters (e.g., communities in social networks), the strongest nodes
# (e.g., search engines), and so on.

# Edges can have a weight, which is the cost or length of the path.
# Edges can have a type, for example 'is-a' or 'is-part-of'.

def nodes(g):
    return set().union(g, *g.values()) # set((node1, node2))

def dfs(g, n1, f=lambda *e: True, v=set(), d=1e10):
    """ Depth-first search.
        Calls f(n1, n2) on the given node, its adjacent nodes if True, and so on.
    """
    v.add(n1) # visited?
    for n2 in g.get(n1, {}):
        if d and f(n1, n2) and not n2 in v:
            dfs(g, n2, f, v, d-1)
    return v

def bfs(g, n1, f=lambda *e: True, v=set(), d=1e10):
    """ Breadth-first search (spreading activation).
        Calls f(n1, n2) on the given node, its adjacent nodes if True, and so on.
    """
    q = collections.deque([(n1, d)])
    while q:
        n1, d = q.popleft()
        v.add(n1)
        for n2 in g.get(n1, {}):
            if d and f(n1, n2) and not n2 in v:
                q.append((n2, d-1))
    return v

# def visit(n1, n2):
#     print(n1, n2)
# 
# g = {
#     'a': {'b': 1},
#     'b': {'c': 1, 'd': 1, 'a': 1},
#     'c': {'x': 1}
# }
# dfs(g, 'a', visit)
# bfs(g, 'a', visit)

def shortest_paths(g, n1, n2=None):
    """ Returns an iterator of shortest paths,
        where each path is a list of node id's.
    """
    # Dijkstra's algorithm, based on Connelly Barnes' implementation:
    # http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/119466

    q = [(0.0, n1, ())]
    v = set() # visited?
    while q:
        d, n, p = heapq.heappop(q)
        if n not in v:
            v.add(n)
            p += (n,)
            if n2 == None and n1 != n:
                yield p
            if n2 != None and n2 == n:
                yield p
                return
            for n, w in g.get(n, {}).items(): # {n1: {n2: cost}}
                if n not in v:
                    heapq.heappush(q, (d + w, n, p))

def shortest_path(g, n1, n2):
    """ Returns the shortest path from n1 to n2.
    """
    try:
        return next(shortest_paths(g, n1, n2))
    except StopIteration:
        return None

sp = shortest_path

# g = {
#     'a': {'b': 1, 'x': 1},  # A -> B -> C -> D
#     'b': {'c': 1},          #   –> X ––––––>
#     'c': {'d': 1},
#     'x': {'d': 1}
# }
# print(sp(g, 'a', 'd'))

def betweenness(g, k=1000):
    """ Returns a dict of node id's and their centrality score (0.0-1.0),
        which is the amount of shortest paths that pass through a node.
    """
    n = nodes(g)
    w = collections.Counter(n)
    if k:
        n = shuffled(n)
        n = list(n)[:k]
    for n in n:
        for p in shortest_paths(g, n):
            for n in p[1:-1]:
                w[n] += 1
 
    # Normalize 0.0-1.0:
    m = max(w.values())
    m = float(m) or 1
    w = {n: w[n] / m for n in w}
    return w

def pagerank(g, iterations=100, damping=0.85, tolerance=0.00001):
    """ Returns a dict of node id's and their centrality score (0.0-1.0),
        which is the amount of indirect incoming links to a node.
    """
    n = nodes(g)
    w = dict.fromkeys(n, 1.0 / (len(n) or 1))
    for i in range(iterations):                                    #       A -> B -> C
        p = w.copy() # prior pagerank                              #      0.3  0.3  0.3
        for k1 in n:                                               # i=0  0.3  0.6  0.6
            for k2 in g.get(k1, {}):                               # i=1  0.3  0.9  1.2
                w[k2] += damping * p[k1] * g[k1][k2] / len(g[k1])  # i=2  0.3  1.2  2.1
            w[k1] += 1 - damping                                   # ...
        m = 0 # normalize
        e = 0 # converged?
        for k in n:
            m += w[k] ** 2.0
        for k in n:
            w[k] /= m ** 0.5 or 1
        for k in n:
            e += abs(w[k] - p[k])
        if e <= len(n) * tolerance:
            break

    return w

def cliques(g):
    """ Returns an iterator of maximal cliques,
        where each clique is a set of node id's
        that are all connected to each other.
    """

    # Bron-Kerbosch's backtracking algorithm 
    # for undirected graphs (A -> B, B -> A).
    def search(r, p, x):
        if p or x:
            u = p | x
            u = u.pop() # pivot
            for n in p - set(g[u]):
                for c in search( 
                  r | set((n,)), 
                  p & set(g[n]), 
                  x & set(g[n])):
                    yield c
                p.remove(n)
                x.add(n)
        else:
            yield r

    return search(set(), set(g), set())

# g = {
#     'a': dict.fromkeys(('b', 'c'), 1),  #    A
#     'b': dict.fromkeys(('a', 'c'), 1),  #  /   \
#     'c': dict.fromkeys(('a', 'b'), 1),  # B ––– C   X
#     'x': {}
# }
# print(list(cliques(g)))

def communities(g, k=4):
    """ Returns an iterator of (overlapping) communities, largest-first,
        where each community is a set of densely connected nodes.
    """
    a = []
    for c1 in cliques(g):
        if len(c1) >= k:
            for c2 in a:
                if len(c1 & c2) >= k - 1: # clique percolation
                    c2.update(c1)
            a.append(c1)
    return reversed(sorted(a, key=len))

def components(g):
    """ Returns an iterator of components, largest-first,
        where each component is a set of connected nodes.
    """
    n = nodes(g)
    a = [set((n,)) | set(g.get(n, ())) for n in n]
    for i in reversed(range(len(a))):
        for j in reversed(range(i + 1, len(a))):
            if a[i] & a[j]: # subsets intersect?
                a[i].update(a[j])
                a.pop(j)
    return reversed(sorted(a, key=len))

# g = {
#     'a': {'b': 1}, # A -> B -> C   X
#     'b': {'c': 1},
#     'x': {}
# }
# print(list(components(g)))

def nameddefaulttuple(name, fields, **default):
    """ Returns a namedtuple with default values.
    """
    r = collections.namedtuple(name, fields)
    r.__new__.__defaults__ = tuple(default[f] for f in fields if f in default)
    return r

# Point = nameddefaulttuple('Point', ('x', 'y', 'z'), z=0)

class Edge(nameddefaulttuple('Edge', ('node1', 'node2', 'weight', 'type'), weight=1.0, type=None)):

    __slots__ = () # disable __dict__ to save memory

    # Algorithms for shortest paths and centrality
    # use an adjacency matrix, i.e., {n1: {n2: w},
    # where w is the weight of the edge n1 -> n2. 

    # But we want {n1: n2: Edge}, so that an edge 
    # can store other metadata besides its weight.
    # We make Edge act as Edge.weight when used in
    # arithmetic operations (x + Edge, x * Edge): 

    __add__ = __radd__ = lambda e, x: x + e.weight
    __sub__ = __rsub__ = lambda e, x: x - e.weight
    __mul__ = __rmul__ = lambda e, x: x * e.weight
    __div__ = __rdiv__ = lambda e, x: x / e.weight
    __float__          = lambda e   :     e.weight

# e = Edge('Garfield', 'cat', weight=1.0, type='is-a')
# 
# print(e)
# print(e.node1)
# print(e.node2)
# print(e.weight)
# print(e.type)
# print(e + 1)
# print(e * 10)
# 
# n1, n2, w, _ = e
# print(n1)
# print(n2)

class Graph(dict): # { node id1: { node id2: edge }}

    def __init__(self, adjacency={}, directed=False):
        self._directed = directed
        for n1 in adjacency:
            for n2 in adjacency[n1]:
                self.add(n1, n2, adjacency[n1][n2])

    @property
    def directed(self):
        return self._directed

    @property
    def density(self):
        n = len(list(self.nodes)) or 1
        e = len(list(self.edges))
        return float(self.directed + 1) * e / n / (n - 1 or 1) # 0.0-1.0

    @property
    def nodes(self):
        """ Returns an iterator of nodes.
        """
        return iter(self)

    @property
    def edges(self):
        """ Returns an iterator of edges,
            each a named tuple (node1, node2, weight, type).
        """
        return iter(set().union(*(e.values() for e in self.values())))

    def edge(self, n1, n2):
        """ Returns the edge from node n1 to n2, or None.
        """
        return self.get(n1, {}).get(n2)

    def incident(self, n):
        """ Returns the edges to and from the given node.
        """
        a = set()
        for n1, n2 in self.items():
            if n == n1:
                a.update(n2.values())
            if n in n2:
                a.add(n2[n])
        return a

    def adjacent(self, n):
        """ Returns the nodes connected to the given node.
        """
        a = set()
        for n1, n2 in self.items():
            if n == n1:
                a.update(n2.keys())
            if n in n2:
                a.add(n1)
        return a

    def degree(self, n):
        return len(self.incident(n))

    def copy(self):
        g = Graph({}, self.directed)
        g.update(self)
        return g

    def update(self, g):
        for n in g.nodes:
            self.add(n)
        for e in g.edges:
            self.add(*e)
        return self

    def add(self, n1, n2=None, weight=1.0, type=None):
        if n2 == None:
            self.setdefault(n1, {})
        if n2 != None:
            self.setdefault(n1, {})
            self.setdefault(n2, {})
            self[n1][n2] = e = Edge(n1, n2, float(weight), type)
        if n2 != None and not self.directed:
            self[n2][n1] = e

    def pop(self, n1, n2=None, default=None):
        if n2 == None:
            for n in self:
                self[n].pop(n1, None)       # n1 <- ...
            v = dict.pop(self, n1, default) # n1 -> ...
        if n2 != None:
            v = self.get(n1, {}).pop(n2, default)
        if n2 != None and not self.directed:
            v = self.get(n2, {}).pop(n1, default)
        return v

    def sub(self, nodes=[]):
        """ Returns a graph with the given nodes, and connecting edges.
        """
        g = Graph({}, self.directed)
        for n in self.nodes:
            if n in nodes:
                g.add(n)
        for e in self.edges:
            if e.node1 in g and e.node2 in g:
                g.add(*e)
        return g

    def nn(self, n, depth=1):
        """ Returns a graph with node n (depth=0),
            nodes connected to this node (depth=1), 
            nodes connected to these nodes (depth=2), ...
        """
        g = Graph({}, self.directed)
        g.add(n)
        for _ in range(depth):
            for e in (e for e in self.edges if not e in g and e.node1 in g or e.node2 in g):
                g.add(*e)
        return g

    def sp(self, n1, n2):
        """ Returns the shortest path from n1 to n2.
        """
        return shortest_path(self, n1, n2)

    def hops(self, n1, n2):
        return len(self.sp(n1, n2)) - 2

    def __contains__(self, v):
        if isinstance(v, Edge):
            return self.edge(v.node1, v.node2) is not None # XXX ignores weight & type
        else:
            return dict.__contains__(self, v)

    def __or__(self, g):
        return union(self, g)

    def __and__(self, g):
        return intersection(self, g)

    def __sub__(self, g):
        return difference(self, g)

    def save(self, path):
        # GraphML: http://graphml.graphdrawing.org
        from xml.sax.saxutils import escape
        s = u'<?xml version="1.0" encoding="utf-8"?>'
        s += '\n<graphml>'
        s += '\n<key for="node" id="label" attr.name="label" attr.type="string"/>'
        s += '\n<key for="edge" id="weight" attr.name="weight" attr.type="double"/>'
        s += '\n<key for="edge" id="type" attr.name="type" attr.type="string"/>'
        s += '\n<graph edgedefault="%sdirected">'   % ('un', '')[self.directed]
        for n in self.nodes:
            s += '\n<node id="%s">'                 % escape(n)
            s += '\n\t<data key="label">%s</data>'  % escape(n)
            s += '\n</node>'
        for e in self.edges:
            s += '\n<edge source="%s" target="%s">' % (escape(e.node1), escape(e.node2))
            s += '\n\t<data key="weight">%s</data>' % (e.weight)
            s += '\n\t<data key="type">%s</data>'   % (e.type or '')
            s += '\n</edge>'
        s += '\n</graph>'
        s += '\n</graphml>'
        f = io.open(path, 'w', encoding='utf-8')
        f.write(s)
        f.close()

    @classmethod
    def load(cls, path):
        t = ElementTree
        t = t.parse(path)
        t = t.find('graph')
        d = t.get('edgedefault') == 'directed'
        g = cls(directed=d)
        for n in t.iter('node'): 
            g.add(unescape(n.get('id')))
        for e in t.iter('edge'): 
            g.add(unescape(e.get('source')), 
                  unescape(e.get('target')), 
                  e.findtext('*[@key="weight"]', 1.0),
                  e.findtext('*[@key="type"]') or None
            )
        return g

#---- GRAPH CONNECTIVITY --------------------------------------------------------------------------

def leaves(g):
    """ Returns the nodes with degree 1.
    """
    return set(n for n in g if len(g[n]) <= 1 and g.degree(n) == 1)

def prune(g, degree=1, weight=0):
    """ Returns a graph with nodes and edges of given degree and weight (or up).
    """
    g = g.copy()
    for e in [e for e in g.edges if e.weight < weight]:
        g.pop(*e[:2])
    for n in [n for n in g.nodes if g.degree(n) < degree]:
        g.pop(n)
    return g

def union(A, B):
    """ Returns a graph with nodes that are in A or B.
    """
    g = Graph({}, A.directed) # A | B
    g.update(A)
    g.update(B)
    return g

def intersection(A, B):
    """ Returns a graph with nodes that are in A and B.
    """
    g = Graph({}, A.directed) # A & B
    for n in A.nodes:
        if n in B is True:
            g.add(n)
    for e in A.edges:
        if e in B is True:
            g.add(*e)
    return g

def difference(A, B):
    """ Returns a graph with nodes that are in A but not in B.
    """
    g = Graph({}, A.directed) # A - B
    for n in A.nodes:
        if n in B is False:
            g.add(n)
    for e in A.edges:
        if e in B is False:
            g.add(*e)
    return g

# g = Graph(directed=False)
# 
# g.add('a', 'b', weight=1)  #     1       1       1       2    
# g.add('b', 'c', weight=1)  # A <---> B <---> C <---> D <---> X
# g.add('c', 'd', weight=1)  #   <--------------------------->  
# g.add('a', 'x', weight=2)  #                 2                
# g.add('d', 'x', weight=2)
# g.add('o')
# 
# print(top(betweenness(g)))
# 
# print(top(pagerank(g)))
# 
# for n1, n2 in nwise(g.sp('d', 'a')):  # DCBA = 3, DXA = 4 
#     print(n1, '->', n2)
# 
# # a has 2 connections:
# for n in g.nn('a', 1):
#     print(n)
# 
# # g has two disconnected subgraphs:
# for g in map(g.sub, components(g)):
#     print(list(g.nodes))

#---- GRAPH VISUALIZATION -------------------------------------------------------------------------
# The graph visualization algorithm repulses any two nodes (k1) and attracts connected nodes (k2),
# by a given force (k ≈ size), with a given velocity magnitude (m ≈ speed).

def layout(g, iterations=200, k1=1.0, k2=1.0, k=1.0, m=1.0, callback=None):
    """ Returns a dict with {node: (x, y)} coordinates.
    """
    n = list(nodes(g))
    x = list(random.random() for _ in n)
    y = list(random.random() for _ in n)
    v = list([0.0, 0.0]      for _ in n) # velocity
    k = k * 1e+2
    m = m * 1e-1

    for _ in range(iterations):
        for i in range(len(n)):
            for j in range(i + 1, len(n)):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx ** 2
                d2+= dy ** 2
                d  = d2 ** 0.5 # distance
                f1 = 0
                f2 = 0
                # 1) Repulse nodes (Coulomb's law)
                if d < k * 10:
                    f1 = k1 * k ** 2 / d2 / 20
                # 2) Attract nodes (Hooke's law)
                if n[i] in g and n[j] in g[n[i]] \
                or n[j] in g and n[i] in g[n[j]]:
                    f2 = k2 * (d2 - k ** 2) / k / d
                # 3) Update velocity:
                vx = dx * (f1 - f2) * m
                vy = dy * (f1 - f2) * m
                v[i][0] += vx
                v[j][0] -= vx
                v[i][1] += vy
                v[j][1] -= vy

        for i in range(len(n)):
            x[i] += min(max(v[i][0], -10), +10)
            y[i] += min(max(v[i][1], -10), +10)
            v[i]  = [0.0, 0.0]

        if callback:
            callback(n, x, y)

    p = zip(x, y)
    p = zip(n, p)
    p = dict(p)
    return p

def zoom(p=[], radius=1.0):
    """ Returns an (x, y)-list scaled to the given radius.
    """
    def polar(x, y):
        return math.sqrt(x * x + y * y), \
               math.atan2(y, x), 
    def cartesian(d, a):
        return math.cos(a) * d, \
               math.sin(a) * d

    p = [polar(x, y) for x, y in p]
    r = radius / max(d for d, a in p)
    p = [cartesian(d * r, a) for d, a in p]
    return p

# layout = {
#     1: (0, 0),
#     2: (1,-1),
#     3: (2, 0)
# }
# 
# print(dict(zip(layout, zoom(layout.values()))))

def visualize(g, **kwargs):
    """ Returns a string with a HTML5 <canvas> element,
        that renders the given graph using a force-directed layout.
    """
    a = {}
    for e in g.edges:
        a.setdefault(e.node1, {})[e.node2] = e.weight

    f = lambda k, v: json.dumps(kwargs.get(k, v))
    s = '\n'.join((
        '<canvas id=%(id)s width=%(width)s height=%(height)s></canvas>',
        '<script src=%(src)s></script>',
        '<script>',
        '\tvar adjacency = %s;' % json.dumps(a).replace('%', '%%'),
        '',
        '\tvar canvas;',
        '\tcanvas = document.getElementById(%(id)s);',
        '\tcanvas.graph = new Graph(adjacency);',
        '\tcanvas.graph.animate(canvas, %(n)s, {',
        '\t\tdirected    : %s,' % f('directed', g.directed),
        '\t\tlabeled     : %s,' % f('labeled', True),
        '\t\tfont        : %s,' % f('font', '10px sans-serif'),
        '\t\tfill        : %s,' % f('fill', '#fff'),
        '\t\tstroke      : %s,' % f('stroke', '#000'),
        '\t\tstrokewidth : %s,' % f('strokewidth', 0.5),
        '\t\tradius      : %s,' % f('radius', 4.0),
        '\t\tk1          : %s,' % f('k1', 1.0),
        '\t\tk2          : %s,' % f('k2', 1.0),
        '\t\tk           : %s,' % f('k', 1.0),
        '\t\tm           : %s'  % f('m', 1.0),
        '\t});',
        '</script>'
    ))
    k = {}
    k.update({'src': 'graph.js', 'id': 'g', 'width': 720, 'height': 480, 'n': 1000})
    k.update(kwargs)
    k = {k: json.dumps(v) for k, v in k.items()}
    return s % k

viz = visualize

###################################################################################################

def setup(src='https://github.com/textgain/grasp/blob/master/', language=('en',)):

    if not os.path.exists(cd('lm')):            # language model
        os.mkdir(cd('lm'))
    if not os.path.exists(cd('kb')):            # knowledge base
        os.mkdir(cd('kb'))

    a = []
    for k in language:
        a.append(('lm',  '~lang.json.zip' % k)) # language detection
        a.append(('lm', '%s-pos.json.zip' % k)) # parts-of-speech
        a.append(('lm', '%s-pov.json'     % k)) # point-of-view
        a.append(('lm', '%s-doc.json'     % k)) # document frequency
        a.append(('kb', '%s-loc.json'     % k)) # locale
        a.append(('kb', '%s-loc.csv'      % k)) # locale
        a.append(('kb', '%s-per.csv'      % k)) # gender
    for f in a:
        if not os.path.exists(cd(*f)):
            try:
                s = os.path.join(src, f[1])
                s = download(s)
                f = open(cd(*f), 'wb')
                f.write(s)
                f.close()
            except NotFound:
                pass
