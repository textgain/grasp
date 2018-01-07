# encoding: utf-8

##### GRASP.PY ####################################################################################

__version__   =  '1.0'
__license__   =  'BSD'
__credits__   = ['Tom De Smedt', 'Guy De Pauw', 'Walter Daelemans']
__email__     =  'info@textgain.com'
__author__    =  'Textgain'
__copyright__ =  'Textgain'

###################################################################################################

# Copyright (c) 2016, Textgain BVBA
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation and/or 
#    other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR 
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
# ETC                               recipes for functions, strings, lists, ...

# Grasp.py is based on the Pattern toolkit (https://github.com/clips/pattern), focusing on brevity.
# Most functions have around 10 lines of code, and most algorithms have around 25-50 lines of code.
# Most classes have about 50-75 lines of code.
###################################################################################################

import sys
import os
import re
import inspect
import logging
import traceback
import threading
import multiprocessing
import multiprocessing.pool
import itertools
import collections
import unicodedata
import codecs
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
import glob
import time
import datetime
import random
import math

from heapq import heappush
from heapq import heappop

try:
    # 3 decimal places (0.001)
    json.encoder.FLOAT_REPR = lambda f: format(f, '.3f')
except:
    pass

PY2 = sys.version.startswith('2')
PY3 = sys.version.startswith('3')

if PY3:
    str, unicode, basestring = bytes, str, str

if PY3:
    # Python 3.4+
    import collections.abc
else:
    # Python 2.7
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
    import urllib.request as urllib2
    import urllib.parse as urlparse
    URLError, Request, urlopen, urlencode, urldecode, urlquote = (
        urllib.error.URLError,
        urllib2.Request,
        urllib2.urlopen,
        urllib.parse.urlencode,
        urllib.parse.unquote,
        urllib.parse.quote
    )
else:
    import urllib2
    import urlparse
    URLError, Request, urlopen, urlencode, urldecode, urlquote = (
        urllib2.URLError,
        urllib2.Request,
        urllib2.urlopen,
        urllib.urlencode,
        urllib.unquote,
        urllib.quote
    )

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
    if sys.version.startswith('2'):
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

def asynchronous(f, callback=lambda v, e: None, daemon=True):
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
        t.daemon = daemon # False = program only ends if thread stops.
        t.start()
        return t
    return thread

# def ping(v, e=None):
#     if e: 
#         raise e
#     print(v)
# 
# pow = asynchronous(pow, ping)
# pow(2, 2)
# pow(2, 3) #.join(1)
# 
# for i in range(10):
#     time.sleep(0.1)
#     print('...')

# Atomic operations are thread-safe, e.g., dict.get() or list.append(),
# but not all operations are atomic, e.g., dict[k] += 1 needs a lock.

lock = threading.RLock()

def atomic(f):
    """ The @atomic decorator executes a function thread-safe.
    """
    def decorator(*args, **kwargs):
        with lock:
            return f(*args, **kwargs)
    return decorator

# hits = collections.defaultdict(int)
# 
# @atomic
# def hit(k):
#     hits[k] += 1

MINUTE, HOUR, DAY = 60, 60*60, 60*60*24

def scheduled(interval=MINUTE):
    """ The @scheduled decorator executes a function periodically (async).
    """
    def decorator(f):
        def timer():
            while 1:
                time.sleep(interval)
                f()
        t = threading.Thread(target=timer)
        t.start()
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
# models['en'] = lambda: Perceptron('huge.json')

###################################################################################################

#---- LOG -----------------------------------------------------------------------------------------
# Functions that access the internet must report the visited URL using the standard logging module.
# See also: https://docs.python.org/2/library/logging.html#logging.Formatter

SIGNED = '%(asctime)s %(filename)s:%(lineno)s %(funcName)s: %(message)s' # 12:59:59 grasp.py#1000

log = logging.getLogger(__name__)
log.level = logging.DEBUG

if not log.handlers:
    log.handlers.append(logging.NullHandler())

def debug(file=sys.stdout, format=SIGNED, date='%H:%M:%S'):
    """ Writes the log to the given file-like object.
    """
    h1 = getattr(debug, '_handler', None)
    h2 = file
    if h1 in log.handlers:
        log.handlers.remove(h1)
    if hasattr(h2, 'write') and hasattr(h2, 'flush'):
        h2 = logging.StreamHandler(h2)
        h2.formatter = logging.Formatter(format, date)
    if isinstance(h2, logging.Handler):
        log.handlers.append(h2)
        debug._handler = h2

# debug()
# debug(open(cd('log.txt'), 'a'))
# request('https://textgain.com')
# debug(False)

###################################################################################################

#---- UNICODE -------------------------------------------------------------------------------------
# The u() function returns a Unicode string (Python 2 & 3).
# The b() function returns a byte string, encoded as UTF-8.

# We use u() as early as possible on all input (e.g. HTML).
# We use b() on URLs.

def u(v, encoding='utf-8'):
    """ Returns the given value as a Unicode string.
    """
    if isinstance(v, str):
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
    if isinstance(v, str):
        return v
    return (u'%s' % v).encode()

#---- ITERATION -----------------------------------------------------------------------------------

def slice(a, *ijn):
    """ Returns an iterator of values from index i to j, by step n.
    """
    return list(itertools.islice(a, *ijn))

def shuffled(a):
    """ Returns an iterator of values in the list, in random order.
    """
    for v in sorted(a, key=lambda v: random.random()):
        yield v

def unique(a):
    """ Returns an iterator of unique values in the list, in order.
    """
    s = set() # seen?
    return iter(v for v in a if not (v in s or s.add(v)))

def chunks(a, n=2):
    """ Returns an iterator of tuples of n consecutive values.
    """
    return zip(*(a[i::n] for i in range(n)))

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

def choice(a, p=[]):
    """ Returns a random element from the given list,
        with optional (non-negative) probabilities.
    """
    p = list(p)
    n = sum(p)
    x = random.uniform(0, n)
    if n == 0:
        return random.choice(a)
    for v, w in zip(a, p):
        x -= w
        if x <= 0:
            return v

# f = {'a': 0, 'b': 0}
# for i in range(100):
#     v = choice(['a', 'b'], p=[0.9, 0.1])
#     f[v] += 1
# 
# print(f)

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

class CSV(table):

    def __init__(self, name='', separator=',', rows=[]):
        """ Returns the given .csv file as a list of rows, each a list of values.
        """
        try:
            self.name      = name
            self.separator = separator
            self._load()
        except IOError:
            pass # doesn't exist (yet)
        if rows:
            self.extend(rows)

    def _load(self):
        with open(self.name, 'r') as f:
            for r in csvlib.reader(f, delimiter=self.separator):
                r = [u(v) for v in r]
                self.append(r)

    def save(self, name=''):
        a = []
        for r in self:
            r = ('"' + u(s).replace('"', '""') + '"' for s in r)
            r = self.separator.join(r)
            a.append(r)
        f = codecs.open(name or self.name, 'w', encoding='utf-8')
        f.write('\n'.join(a))
        f.close()

    def update(self, rows=[], index=0):
        """ Appends the rows that have no duplicates in the given column(s).
        """
        u = set(map(repr, self[:,index])) # unique + hashable slices (slow)
        for r in rows:
            k = repr(r[index])
            if k not in u:
                self.append(r)
                u.add(k)

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

def cd(*args):
    """ Returns the directory of the script that calls cd() + given relative path.
    """
    f = inspect.currentframe()
    f = inspect.getouterframes(f)[1][1] 
    f = f != '<stdin>' and f or os.getcwd()
    p = os.path.realpath(f)
    p = os.path.dirname(p)
    p = os.path.join(p, *args)
    return p

# print(cd('test.csv'))

# for code, state, adj, region, gov, city, lang, rating in csv(cd('loc.csv')):
#     print(state)

#---- SQL -----------------------------------------------------------------------------------------
# A database is a collection of tables, with rows and columns of structured data.
# Rows can be edited or selected with SQL statements (Structured Query Language).
# Rows can be indexed for faster retrieval or related to other tables.

# SQLite is a lightweight engine for a portable database stored as a single file.

# https://www.sqlite.org/datatype3.html
AFFINITY = collections.defaultdict(
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
        with indices on fields ending with '*'.
    """
    s = 'create table if not exists `%s` (' % table + 'id integer primary key);'
    i = 'create index if not exists `%s_%s` on `%s` (`%s`);'
    for k in fields:
        k = re.sub(r'\*$', '', k)  # 'name*' => 'name'
        v = AFFINITY[type.get(k)]  #     str => 'text'
        s = s[:-2] + ', `%s` %s);' % (k, v)
    for k in fields:
        if k.endswith('*'):
            s += i % ((table, k[:-1]) * 2)
    return s

# print(schema('persons', 'name*', 'age', age=int))

class DatabaseError(Exception):
    pass

class Database(object):

    def __init__(self, name, schema=None, timeout=10, factory=sqlite.Row):
        """ SQLite database interface.
        """
        self.connection = sqlite.connect(name, timeout)
        self.connection.row_factory = factory
        if schema:
            for q in schema.split(';'):
                self(q, commit=False)
            self.commit()

    def __call__(self, sql, values=(), first=False, commit=True):
        """ Executes the given SQL statement.
        """
        try:
            r = self.connection.cursor().execute(sql, values)
            if commit:
                self.connection.commit()
            if first:
                return r.fetchone() if r else r  # single row
            else:
                return r
        except Exception as e:
            raise DatabaseError(str(e))

    def execute(self, *args, **kwargs):
        return self(*args, **kwargs)

    def commit(self):
        return self.connection.commit()

    def rollback(self):
        return self.connection.rollback()

    @property
    def id(self):
        return self('select last_insert_rowid()').fetchone()[0]

    def find(self, table, *fields, **filters):
        return self(*SELECT(table, *fields, **filters))

    def first(self, table, *fields, **filters):
        return self(*SELECT(table, *fields, **filters), first=True)

    def append(self, table, **fields):
        return self(*INSERT(table, **fields), 
                        commit=fields.pop('commit', True))

    def update(self, table, id, **fields):
        return self(*UPDATE(table, id, **fields), 
                        commit=fields.pop('commit', True))

    def remove(self, table, id):
        return self(*DELETE(table, id), 
                        commit=fields.pop('commit', True))

    def __del__(self):
        try: 
            self.connection.commit()
            self.connection.close()
            self.connection = None
        except:
            pass

# db = Database(cd('test.db'), schema('persons', 'name*', 'age', age=int))
# db.append('persons', name='Tom', age=30)
# db.append('persons', name='Guy', age=30)
# 
# for id, name, age in db.find('persons', age='>20'):
#     print(name, age)

def concat(a, format='%s', separator=', '):
  # concat([1, 2, 3]) => '1, 2, 3'
    return separator.join(format % v for v in a)

def SELECT(table, *fields, **where):
    """ Returns an SQL SELECT statement + parameters.
    """

    def op(v):
        if isinstance(v, basestring) and re.search(r'^<=|>=', v): # '<=10'
            return v[:2], v[2:]
        if isinstance(v, basestring) and re.search(r'^<|>', v): # '<10'
            return v[:1], v[1:]
        if isinstance(v, basestring) and re.search(r'\*', v): # '*ly'
            return 'like', v.replace('*', '%')
        if hasattr(v, '__iter__'):
            return 'in', v
        else:
            return '=', v

    s = 'select %s from %s where %s ' + 'limit %i, %i order by `%s`;' % (
         where.pop('slice', (0, -1)) + (
         where.pop('sort', 'id'),)
    )
    f = concat(fields or '*')
    k = where.keys()    # ['name', 'age']
    v = where.values()  # ['Tom*', '>10']
    v = map(op, v)      # [('like', 'Tom%'), ('>', '10')]
    v = zip(*v)         #  ('like', '>'), ('Tom%', '10')
    v = iter(v)
    q = next(v, ())
    v = next(v, ())
    s = s % (f, table, concat(zip(k, q), '`%s` %s ?', 'and'))
    s = s.replace('limit 0, -1 ', '', 1)
    s = s.replace('where  ', '', 1)
    return s, tuple(v)

# print(SELECT('persons', '*', age='>10', slice=(0, 10)))

def INSERT(table, **fields):
    """ Returns an SQL INSERT statement + parameters.
    """
    s = 'insert into `%s` (%s) values (%s);'
    k = fields.keys()
    v = fields.values()
    s = s % (table, concat(k, '`%s`'), concat('?' * len(v)))
    return s, tuple(v)

# print(INSERT('persons', name='Smith', age=10))

def UPDATE(table, id, **fields):
    """ Returns an SQL UPDATE statement + parameters.
    """
    s = 'update `%s` set %s where id=?;'
    k = fields.keys()
    v = fields.values()
    s = s % (table, concat(k, '`%s`=?'))
    s = s.replace(' set  ', '', 1)
    return s, tuple(v) + (id,)

# print(UPDATE('persons', 1, name='Smith', age=20))

def DELETE(table, id):
    """ Returns an SQL DELETE statement + parameters.
    """
    s = 'delete from `%s` where id=?;' % table
    return s, (id,)

# print(DELETE('persons' 1))

#---- ENCRYPTION ----------------------------------------------------------------------------------
# The pw() function is secure enough for storing passwords; encrypt() and decrypt() are not secure.

def key(n=32):
    """ Returns a new key of length n.
    """
    k = os.urandom(256)
    k = binascii.hexlify(k)[:n]
    return u(k)

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
    k = base64.b64encode(os.urandom(32))
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
        eq = ch1 == ch2 # contstant-time comparison
    return eq

# print(pw_ok('1234', pw('1234')))

##### ML ##########################################################################################

#---- MODEL --------------------------------------------------------------------------------------
# The Model base class is inherited by Perceptron, Bayes, ...

class Model(object):

    def __init__(self, examples=[], **kwargs):
        self.labels = {}

    def train(self, v, label=None):
        raise NotImplementedError

    def predict(self, v):
        raise NotImplementedError

    def save(self, f):
        json.dump(self.__dict__, f)

    @classmethod
    def load(cls, f):
        self = cls()
        for k, v in json.load(f).items():
            try:
                getattr(self, k).update(v) # defaultdict?
            except:
                setattr(self, k, v)
        return self

#---- PERCEPTRON ----------------------------------------------------------------------------------
# The Perceptron or single-layer neural network is a supervised machine learning algorithm.
# Supervised machine learning uses labeled training examples to infer statistical patterns.
# Each example is a set of features – e.g., set(('lottery',)) – and a label (e.g., 'spam').

# The Perceptron takes a list of examples and learns what features are associated with what labels.
# The resulting 'model' can then be used to predict the label of new examples.

def avg(a):
    a = list(a)
    n = len(a) or 1
    s = sum(a)
    return float(s) / n

def sd(a):
    a = list(a)
    n = len(a) or 1
    m = avg(a)
    return math.sqrt(sum((v - m) ** 2 for v in a) / n)

def iavg(x, m=0.0, sd=0.0, t=0):
    """ Returns the iterative (mean, standard deviation, number of samples).
    """
    t += 1
    v  = sd ** 2 + m ** 2 # variance
    v += (x ** 2 - v) / t
    m += (x ** 1 - m) / t
    sd = math.sqrt(v - m ** 2)
    return (m, sd, t)

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

def top(p):
    """ Returns a (key, value)-tuple with the max value in the dict.
    """
    if p:
        v = max(p.values())
    else:
        v = 0.0
    k = [k for k in p if p[k] == v]
    k = random.choice(k)
    return k, v

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

        for i in range(n):
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
            for f in v:
                # Error correction:
                cumsum(label, f, +1, self._t)
                cumsum(guess, f, -1, self._t)
            self._t += 1

        self._p = iavg(abs(p), *self._p) # (mean, sd, t)

    def predict(self, v, normalize=True):
        """ Returns a dict of (label, probability)-items.
        """
        p = dict.fromkeys(self.labels, 0.0)
        t = float(self._t)
        for label, features in self.weights.items():
            n = 0
            for f in v:
                if f in features:
                    w = features[f]
                    n = n + (w[1] + w[0] * (t - w[2])) / t
            p[label] = n

        if normalize:
            # 1. Divide values by avg + sd (-1 => +1)
            # 2. Softmax to values between 0.1 => 0.9
            #    (with < 0.1 and > 0.9 for outliers)
            p = softmax(p, a=(self._p[0] + self._p[1]))
        return p

# p = Perceptron(examples=[
#     (('woof', 'bark'), 'dog'),
#     (('meow', 'purr'), 'cat')], n=10)
# 
# print(p.predict(('meow',)))
# 
# p.save(open('model.json', 'w'))
# p = Perceptron.load(open('model.json'))

#---- NAIVE BAYES ---------------------------------------------------------------------------------
# The Naive Bayes model is a simple alternative for Perceptron (it trains very fast).
# It is based on the likelihood that a given feature occurs with a given label.

# The probability that something big and bad is a wolf is: 
# p(big|wolf) * p(bad|wolf) * p(wolf) / (p(big) * p(bad)). 

# So it depends on the frequency of big wolves, bad wolves,
# other wolves, other big things, and other bad things.

class Bayes(Model):
    
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
        for f in v:
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
            w = (self.weights[x].get(f, 0.1) / n for f in v)
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

#---- FEATURES ------------------------------------------------------------------------------------
# Character 3-grams are sequences of 3 successive characters: 'hello' => 'hel', 'ell', 'llo'.
# Character 3-grams are useful as training examples for text classifiers,
# capturing 'small words' such as pronouns, smileys, word suffixes (-ing)
# and language-specific letter combinations (oeu, sch, tch, ...)

URL = re.compile(r'https?://.*?(?=[,.!?)]*(?:\s|$))')
REF = re.compile(r'[\w._\-]*@[\w._\-]+', flags=re.U)

def chngrams(s, n=3):
    """ Returns an iterator of character n-grams.
    """
    for i in range(len(s) - n + 1):
        yield s[i:i+n] # 'hello' => 'hel', 'ell', 'llo'

def ngrams(s, n=2):
    """ Returns an iterator of word n-grams.
    """
    for w in chngrams([w for w in re.split(r'\s+', s) if w], n):
        yield tuple(w)

def v(s, features=('ch3',)): # (vector)
    """ Returns a set of character trigrams in the given string.
        Can be used as Perceptron.train(v(s)) or predict(v(s)).
    """
   #s = s.lower()
    s = re.sub(URL, 'http://', s)
    s = re.sub(REF, '@name', s)
    v = collections.Counter()
    v[''] = 1 # bias
    for f in features:
        if f[0] == 'c': # 'c1' (punctuation, diacritics)
            v.update(chngrams(s, n=int(f[-1])))
        if f[0] == 'w': # 'w1'
            v.update(  ngrams(s, n=int(f[-1])))
    return v

vec = v

# data = []
# for id, username, tweet, date in csv(cd('spam.csv')):
#     data.append((v(tweet), 'spam'))
# for id, username, tweet, date in csv(cd('real.csv')):
#     data.append((v(tweet), 'real'))
# 
# p = Perceptron(examples=data, n=10)
# p.save(open('spam-model.json', 'w'))
# 
# print(p.predict(v('Be Lazy and Earn $3000 per Week'))) # {'real': 0.15, 'spam': 0.85}

#---- FEATURE SELECTION ---------------------------------------------------------------------------
# Feature selection identifies the best features, by evaluating their statistical significance.

def pp(data=[]): # (posterior probability)
    """ Returns a {feature: {label: frequency}} dict
        for the given set of (vector, label)-tuples.
    """
    f1 = collections.defaultdict(float) # {label: count}
    f2 = collections.defaultdict(float) # {feature: count}
    f3 = collections.defaultdict(float) # {feature, label: count}
    p  = {}
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
        for the given set of (vector, label)-tuples.
    """
    from scipy.stats import chi2_contingency as chi2

    f1 = collections.defaultdict(float) # {label: count}
    f2 = collections.defaultdict(float) # {feature: count}
    f3 = collections.defaultdict(float) # {feature, label: count}
    p  = {}
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

def topn(p, n=10, reverse=False):
    """ Returns an iterator of (key, value)-tuples
        ordered by the highest values in the dict.
    """
    for k in sorted(p, key=p.get, reverse=not reverse)[:n]:
        yield k, p[k]

# data = [
#     (set(('yawn', 'meow')), 'cat'),
#     (set(('yawn',       )), 'dog')] * 10
# 
# bias = pp(data)
# 
# for f, p in fsel(data).items():
#     if p < 0.01:
#         print(f)
#         print(top(bias[f]))
#         # 'meow' is significant (always predicts 'cat')
#         # 'yawn' is independent (50/50 'dog' and 'cat')

#---- ACCURACY ------------------------------------------------------------------------------------
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
        m = Model(examples=y, **kwargs)
        f = confusion_matrix(m, test=x)
        for label, n in m.labels.items():
            if not weighted:
                n = 1
            precision, recall = test(f, target=label)
            P.append((precision, n))
            R.append((recall, n))

            if debug:
                # k 1 P 0.99 R 0.99 spam
                print('k %i' % (i+1), 'P %.2f' % precision, 'R %.2f' % recall, label)

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

class calibrate(Model):

    def __init__(self, model, label, data=[]):
        """ Returns a new Model calibrated on the given data,
            which is a set of (vector, label)-tuples.
        """
        self._model = model
        self._label = label
        # Isotonic regression:
        y = ((model.predict(v)[label], label == x) for v, x in data)
        y = sorted(y) # monotonic
        y = zip(*y)
        y = list(y or ((),()))
        x = list(y[0])
        y = list(y[1])
        y = pav(y)
        x = [0] + x + [1]
        y = [0] + y + [1]
        f = {}
        i = 0
        # Linear interpolation:
        for p in range(100 + 1):
            p *= 0.01
            while x[i] < p:
                i += 1
            f[p] = (y[i-1] * (x[i] - p) + y[i] * (p - x[i-1])) / (x[i] - x[i-1])
        self._f = f

    def predict(self, v):
        """ Returns the label's calibrated probability (0.0-1.0).
        """
        p = self._model.predict(v)[self._label]
        p = self._f[round(p, 2)]
        return p

    def save(self, f):
        raise NotImplementedError
    
    @classmethod
    def load(cls, f):
        raise NotImplementedError

# data = []
# for review, polarity in csv('reviews-assorted1000.csv'):
#     data.append((v(review), polarity))
# 
# m = Model.load('sentiment.json')
# m = calibrate(m, '+', data)

#---- VECTOR --------------------------------------------------------------------------------------
# A vector is a {feature: weight} dict, with n features, or n dimensions.

# If {'x1':1, 'y1':2} and {'x2':3, 'y2':4} are two points in 2D,
# then their distance is: sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2).
# The distance can be calculated for points in 3D, 4D, or in nD.

# Another measure of similarity is the angle between two vectors (cosine).
# This works well for text features.

def distance(v1, v2):
    """ Returns the distance of the given vectors.
    """
    return sum(pow(v1.get(f, 0.0) - v2.get(f, 0.0), 2) for f in features((v1, v2))) ** 0.5

def dot(v1, v2):
    """ Returns the dot product of the given vectors.
        Each vector is a dict of (feature, weight)-items.
    """
    return sum(v1.get(f, 0.0) * w for f, w in v2.items())

def norm(v):
    """ Returns the norm of the given vector.
    """
    return sum(w ** 2 for f, w in v.items()) ** 0.5

def cos(v1, v2):
    """ Returns the angle of the given vectors (0.0-1.0).
    """
    return 1 - dot(v1, v2) / (norm(v1) * norm(v2) or 1) # cosine distance

def knn(v, vectors=[], k=3, distance=cos):
    """ Returns the k nearest neighbors from the given list of vectors.
    """
    nn = sorted((1 - distance(v, x), x) for x in vectors)
    nn = reversed(nn)
    nn = list(nn)[:k]
    return nn

def sparse(v):
    """ Returns a vector with non-zero weight features.
    """
    return v.__class__({f: w for f, w in v.items() if w != 0})

def tf(v):
    """ Returns a vector with normalized weights
        (term frequency, sum to 1.0).
    """
    n = sum(v.values())
    n = float(n or 1)
    return v.__class__({f: w / n for f, w in v.items()})

def tfidf(vectors=[]):
    """ Returns an iterator of vectors with normalized weights
        (term frequency–inverse document frequency).
    """
    df = collections.Counter() # stopwords have higher df (the, or, I, ...)
    if not isinstance(vectors, list):
        vectors = list(vectors)
    for v in vectors:
        df.update(v)
    for v in vectors:
        yield v.__class__({f: w / float(df[f] or 1) for f, w in v.items()})

def features(vectors=[]):
    """ Returns the set of features for all vectors.
    """
    return set().union(*vectors)

def index(a=[], inverted=True):
    if inverted:
        return {v: i for i, v in enumerate(a)}
    else:
        return {i: v for i, v in enumerate(a)}

def centroid(vectors=[]):
    """ Returns the mean vector for all vectors.
    """
    v = list(vectors)
    n = float(len(v))
    return {f: sum(v.get(f, 0) for v in v) / n for f in features(v)}

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

# examples = [
#     ("'I know some good games we could play,' said the cat.", 'seuss'),
#     ("'I know some new tricks,' said the cat in the hat."   , 'seuss'),
#     ("They roared their terrible roars"                     , 'sendak'),
#     ("They gnashed their terrible teeth"                    , 'sendak'),
#     
# ]
# 
# v, labels = zip(*examples) # = unzip
# v = list(wc(tokenize(v)) for v in v)
# v = list(tfidf(v))
# 
# labels = {id(v): label for v, label in zip(v, labels)} # { vector id: label }
# 
# x = wc(tokenize('They rolled their terrible eyes'))
# x = wc(tokenize("'Look at me! Look at me now!' said the cat."))
# 
# for w, nn in knn(x, v, k=3):
#     w = round(w, 2)
#     print(w, labels[id(nn)])
# 
# print(majority(labels[id(nn)] for w, nn in knn(x, v, k=3)))

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

#---- VECTOR MATRIX -------------------------------------------------------------------------------

def frank(vectors=[]):
    """ Returns a (feature, index)-dict ranked by document frequency,
        i.e., the feature that occurs in the most vectors has rank 0.
    """
    r = sum(map(collections.Counter, vectors), collections.Counter())
    r = sorted(r, key=lambda f: (-r[f], f))
    r = map(reversed, enumerate(r))
    r = collections.OrderedDict(r)
    return r

# print(rankf([{'a': 0}, {'b': 1}, {'b': 1, 'c': 1}, {'c': 1}]))
# {'b': 0, 'c': 1}

def matrix(vectors=[], features={}):
    """ Returns a 2D numpy.ndarray of the given vectors,
        with the given (feature, index)-dict as columns.
    """
    import numpy
    f = features or frank(vectors)
    m = numpy.zeros((len(vectors), len(f)))
    for v, a in zip(vectors, m):
        a.put(map(f.__getitem__, v), v.values())
    return m

##### NLP #########################################################################################

#---- TEXT ----------------------------------------------------------------------------------------

LINK = re.compile(r'(https?://.*?|www\..*?|[\w|-]+\.(?:com|net|org))(?:[.?!,)])?(?:\'|\"|\s|$)')

def diff(s1, s2):
    raise NotImplementedError

def readability(s):
    """ Returns the readability of the given string (0.0-1.0).
    """
    # Flesch Reading Ease; Farr, Jenkins & Patterson's formula.

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
     u'ø' : 'o' ,
     u'ß' : 'ss',
     u'œ' : 'ae',
     u'æ' : 'oe',
     u'“' : '"' ,
     u'”' : '"' ,
     u'‘' : "'" ,
     u'’' : "'" ,
     u'⁄' : '/' ,
     u'¿' : '?' ,
     u'¡' : '!'}.items():
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
        if re.search(r'(?i)ss|[^s]sis|[^mnotr]us$', w     ):   # ± 93%
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

#---- TOKENIZER -----------------------------------------------------------------------------------
# The tokenize() function identifies tokens (= words, symbols) and sentence breaks in a string.
# The task involves handling abbreviations, contractions, hyphenation, emoticons, URLs, ...

EMOJI = set((
    u'😊', u'☺️', u'😉', u'😌', u'😏', u'😎', u'😍', u'😘', u'😴', u'😀', u'😃', u'😄', u'😅', 
    u'😇', u'😂', u'😭', u'😢', u'😱', u'😳', u'😜', u'😛', u'😁', u'😐', u'😕', u'😧', u'😦', 
    u'😒', u'😞', u'😔', u'😫', u'😩', u'😠', u'😡', u'🙊', u'🙈', u'💔', u'❤️', u'💕', u'♥', 
    u'👌', u'✌️', u'👍', u'🙏'
))

EMOTICON = set((
    ':-D', '8-D', ':D', '8)', '8-)',
    ':-)', '(-:', ':)', '(:', ':-]', ':=]', ':-))',
    ':-(', ')-:', ':(', '):', ':((', ":'(", ":'-(",
    ':-P', ':-p', ':P', ':p', ';-p',
    ':-O', ':-o', ':O', ':o', '8-o'
    ';-)', ';-D', ';)',
    ':-S', ':-s',
    '<3'
))

abbreviations = {
    'en': set(('a.m.', 'cf.', 'e.g.', 'etc.', 'i.e.', 'p.m.', 'vs.', 'w/', 'Dr.', 'Mr.'))
}

contractions = {
    'en': set(("'d", "'m", "'s", "'ll", "'re", "'ve", "n't"))
}

for c in contractions.values():
    c |= set(s.replace("'", u'’') for s in c) # n’t

_RE_EMO1 = '|'.join(re.escape(' '+' '.join(s)) for s in EMOTICON | EMOJI).replace(r'\-\ ', '\- ?')
_RE_EMO2 = '|'.join(re.escape(     ''.join(s)) for s in EMOTICON | EMOJI)

def tokenize(s, language='en', known=[]):
    """ Returns the string with punctuation marks split from words , 
        and sentences separated by newlines.
    """
    p = u'….¡¿?!:;,/(){}[]\'`‘’"“”„&–—'
    p = re.compile(r'([%s])' % re.escape(p))
    f = re.sub

    # Find tokens w/ punctuation (URLs, numbers, ...)
    w  = set(known)
    w |= set(re.findall(r'https?://.*?(?:\s|$)', s))                      # http://...
    w |= set(re.findall(r'(?:\s|^)((?:[A-Z]\.)+[A-Z]\.?)', s))            # U.S.
    w |= set(re.findall(r'(?:\s|^)([A-Z]\. )(?=[A-Z])', s))               # J. R. R. Tolkien
    w |= set(re.findall(r'(\d+[\.,:/″][\d\%]+)', s))                      # 1.23
    w |= set(re.findall(r'(\w+\.(?:doc|html|jpg|pdf|txt|zip))', s, re.U)) # cat.jpg
    w |= abbreviations.get(language, set())                               # etc.
    w |=  contractions.get(language, set())                               # 're
    w  = '|'.join(f(p, r' \\\1 ', w).replace('  ', ' ') for w in w)
    e1 = _RE_EMO1
    e2 = _RE_EMO2

    # Split punctuation:
    s = f(p, ' \\1 ', s) + ' '
    s = f(r'\t', ' ', s)
    s = f(r' +', ' ', s)
    s = f(r'\. (?=\.)', '.', s)                                           #  ...
    s = f(r'(\(|\[) (\.+) (\]|\))', '\\1\\2\\3', s)                       # (...)
    s = f(r'(\(|\[) (\d+) (\]|\))', '\\1\\2\\3', s)                       # [1]
    s = f(r'(^| )(p?p) \. (?=\d)', '\\1\\2. ', s)                         # p.1
    s = f(r'(?<=\d)(mb|gb|kg|km|m|ft|h|hrs|am|pm) ', ' \\1 ', s)          # 5ft
    s = f(u'(^|\s)($|£|€)(?=\d)', '\\1\\2 ', s)                           # $10
    s = f(r'& (#\d+|[A-Za-z]+) ;', '&\\1;', s)                            # &amp; &#38;
    s = f(w , lambda m: ' %s ' % m.group().replace(' ', ''), s)           # e.g. 1.2
    s = f(w , lambda m: ' %s ' % m.group().replace(' ', ''), s)           # U.K.'s
    s = f(e1, lambda m: ' %s ' % m.group().replace(' ', ''), s)           # :-)
    s = f(r'(https?://.*?)([.?!,)])(?=\s|$)', '\\1 \\2', s)               # (http://)
    s = f(r'(https?://.*?)([.?!,)])(?=\s|$)', '\\1 \\2', s)               # (http://),
    s = f(r'www \. (.*?) \. (?=[a-z])', 'www.\\1.', s)                    # www.goo.gl
    s = f(r' \. (com?|net|org|be|cn|de|fr|nl|ru|uk)(?=\s|$)', '.\\1', s)  # google.com
    s = f(r'(\w+) \. (?=\w+@\w+\.(?=com|net|org))', ' \\1.', s)           # a.bc@gmail
    s = f(r'(?<=[a-z])(-)\n([a-z]+(?:\s|$))', '\\2\n', s, re.U)           # be-\ncause
    s = f(r' +', ' ', s)

    # Split sentences:
    s = f(r'\r+', '\n', s)
    s = f(r'\n+', '\n', s)
    s = f(u' ([….?!]) (?!=[….?!])', ' \\1\n', s)                          # .\n
    s = f(u'\n([’”)]) ', ' \\1\n', s)                                     # “Wow.”
    s = f(r'\n((#\w+) ?) ', ' \\1\n', s)                                  #  Wow! #nice
    s = f(r'\n(((%s) ?)+) ' % e2, ' \\1\n', s)                            #  Wow! :-)
    s = f(r' (%s) (?=[A-Z])' % e2, ' \\1\n', s)                           #  Wow :-) The
    s = f(r' (etc\.) (?=[A-Z])', ' \\1\n', s)                             # etc. The
    s = f(r'\n, ', ' , ', s)                                              # Aha!, I said
    s = f(r' +', ' ', s)
    s = s.split('\n')

    # Balance quotes:
    for i in range(len(s)-1):
        j = i + 1
        if s[j].startswith(('" ', "' ")) and \
           s[j].count(s[j][0]) % 2 == 1  and \
           s[i].count(s[j][0]) % 2 == 1:
            s[i] += ' ' + s[j][0]
            s[j] = s[j][2:]

    return '\n'.join(s).strip()

# s = u"RT @user: Check it out... 😎 (https://www.textgain.com) #Textgain cat.jpg"
# s = u"There's a sentence on each line, each a space-separated string of tokens (i.e., words). :)"
# s = u"Title\nA sentence.Another sentence. ‘A citation.’ By T. De Smedt."

def wc(s):
    """ Returns a (word, count)-dict, lowercase.
    """
    f = re.split(r'\s+', s.lower())
    f = collections.Counter(f)
    return f

# print(wc(tokenize('The cat sat on the mat.')))

#---- PART-OF-SPEECH TAGGER -----------------------------------------------------------------------
# Part-of-speech tagging predicts the role of each word in a sentence: noun, verb, adjective, ...
# Different words have different roles according to their position in the sentence (context).
# In 'Can I have that can of soda?', 'can' is used as a verb ('I can') and a noun ('can of').

# We want to train a machine learning algorithm (Percepton) with context vectors as examples,
# where each context vector includes the word, word suffix, and words to the left and right.

# The part-of-speech tag of unknown words is predicted by their suffix (e.g., -ing, -ly)
# and their context (e.g., a determiner is often followed by an adjective or a noun).

def ctx(*w):
    """ Returns the given [token, tag] list parameters as 
        a context vector (i.e., token, tokens left/right) 
        that can be used to train a part-of-speech tagger.
    """
    m = len(w) // 2 # middle
    v = set()
    for i, (w, tag) in enumerate(w):
        i -= m
        if i == 0:
            v.add(' ')                       # bias
            v.add('1 %+d %s' % (i, w[:+1]))  # capitalization
            v.add('* %+d %s' % (i, w[-6:]))  # token
            v.add('^ %+d %s' % (i, w[:+3]))  # token head
            v.add('$ %+d %s' % (i, w[-3:]))  # token suffix
        else:
            v.add('$ %+d %s' % (i, w[-3:]))  # token suffix left/right
            v.add('? %+d %s' % (i, tag   ))  # token tag
    return v

# print(ctx(['The', 'DET'], ['cat', 'NOUN'], ['sat', 'VERB'])) # context of 'cat'
#
# set([
#     ' '        , 
#     '$ -1 The' , 
#     '? -1 DET' , 
#     '1 +0 c'   , 
#     '* +0 cat' , 
#     '^ +0 cat' , 
#     '$ +1 sat' , 
#     '? +1 VERB', 
# ])

@printable
class Sentence(list):

    def __init__(self, s=''):
        """ Returns the tagged sentence as a list of [token, tag]-values.
        """
        if isinstance(s, list):
            list.__init__(self, s)
        if isinstance(s, (str, unicode)) and s:
            for w in s.split(' '):
                w = u(w)
                w = re.split(r'(?<!\\)/', w + '/')[:2]
                w = [w.replace('\/', '/') for w in w]
                self.append(w)

    def __str__(self):
        return ' '.join('/'.join(w.replace('/', '\\/') for w in w) for w in self)

    def __repr__(self):
        return 'Sentence(%s)' % repr(u(self))

# s = 'The/DET cat/NOUN sat/VERB on/PREP the/DET mat/NOUN ./PUNC'
# for w, tag in Sentence(s):
#     print(w, tag)

TAGGER = LazyDict() # {'en': Model}

TAGGER['en'] = lambda: Perceptron.load(open(cd('en-pos.json')))

def tag(s, language='en'):
    """ Returns the tokenized + tagged string.
    """
    return '\n'.join(u(s) for s in parse(s, language))

def parse(s, language='en'):
    """ Returns the tokenized + tagged string,
        as an iterator of Sentence objects.
    """
    model = TAGGER[language]
    s = tokenize(s)
    s = s.replace('/', '\\/')
    s = s.split('\n')
    for s in s:
        a = Sentence()
        for w in nwise(Sentence('  %s  ' % s), n=5):
            if len(a) > 1:
                w[0][1] = a[-2][1] # use predicted tag left
                w[1][1] = a[-1][1]
            tag, p = top(model.predict(ctx(*w)))
            a.append([w[2][0], tag])
        yield a

# for s in parse("We're all mad here. I'm mad. You're mad."):
#     print(repr(s))

PTB = {           # Penn Treebank tagset                                           # EN
    u'NOUN' : set(('NN', 'NNS', 'NNP', 'NNPS', 'NP')),                             # 30%
    u'VERB' : set(('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD')),                # 14%
    u'PUNC' : set(('LS', 'SYM', '.', ',', ':', '(', ')', '``', "''", '$', '#')),   # 11%
    u'PREP' : set(('IN', 'PP')),                                                   # 10%
    u'DET'  : set(('DT', 'PDT', 'WDT', 'EX')),                                     #  9%
    u'ADJ'  : set(('JJ', 'JJR', 'JJS')),                                           #  7%
    u'ADV'  : set(('RB', 'RBR', 'RBS', 'WRB')),                                    #  4%
    u'NUM'  : set(('CD', 'NO')),                                                   #  4%
    u'PRON' : set(('PR', 'PRP', 'PRP$', 'WP', 'WP$')),                             #  3%
    u'CONJ' : set(('CC', 'CJ')),                                                   #  2%
    u'X'    : set(('FW',)),                                                        #  2%
    u'PRT'  : set(('POS', 'PT', 'RP', 'TO')),                                      #  2%
    u'INTJ' : set(('UH',)),                                                        #  1%
}

WEB = dict(PTB, **{
    u'NOUN' : set(('NN', 'NNS', 'NP')),                                            # 14%
    u'NAME' : set(('NNP', 'NNPS', '@')),                                           # 11%
    u'PRON' : set(('PR', 'PRP', 'PRP$', 'WP', 'WP$', 'PR|MD', 'PR|VB', 'WP|VB')),  #  9%
    u'URL'  : set(('URL',)),                       # 'youve'  'thats'  'whats'     #  1%
    u':)'   : set((':)',)),                                                        #  1%
    u'#'    : set(('#',)),                                                         #  1%
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

# corpus = cd('/corpora/wsj.txt')
# corpus = u(open(corpus).read())
# corpus = corpus.split('\n')
# corpus = corpus[:48000] # ~ 1M tokens

# Create context vectors from WSJ sentences, using the simplified universal tagset.

# data = []
# for s in corpus:
#     for w in nwise(Sentence('  %s  ' % s), n=5):
#         w = [universal(*w) for w in w]
#         data.append((ctx(*w), w[2][1]))
# 
# print('%s sentences' % len(corpus))
# print('%s tokens'    % len(data))
# 
# print(kfoldcv(Perceptron, data, k=3, n=5, weighted=True, debug=True)) # 0.96 0.96
# 
# en = Perceptron(data, n=5)
# en.save(open('en.json', 'w'))
# 
# print(tag('What a great day! I love it.'))

#---- PART-OF-SPEECH SEARCH -----------------------------------------------------------------------
# The chunk() function yields parts of a part-of-speech-tagged sentence that match a given pattern.
# For example, 'ADJ* NOUN' yields all nouns in a sentence, and optionally the preceding adjectives.

# The constituents() function yields NP, VP, AP and PP phrases.
# A NP (noun phrase) is a noun + preceding determiners and adjectives (e.g., 'the big black cat').
# A VP (verb phrase) is a verb + preceding auxillary verbs (e.g., 'might be doing'). 

TAG = set((
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
))

inflections = {
    'aux'  : r"can|shall|will|may|must|could|should|would|might|'ll|'d",
    'be'   : r"be|am|are|is|was|were|being|been|'m|'re|'s",
    'have' : r"have|has|had|having|'ve",
    'do'   : r"do|does|did|doing"
}

class Phrase(Sentence):

    def __repr__(self):
        return 'Phrase(%s)' % repr(u(self))

_RE_TAG = '|'.join(map(re.escape, TAG)) # NAME|NOUN|\:\)|...

def chunk(pattern, sentence, replace=[]):
    """ Yields an iterator of matching Phrase objects from the given Sentence.
        The search pattern is a sequence of tokens (talk-, -ing), tags (VERB),
        token/tags (-ing/VERB), escaped characters (:\)), control characters: 
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
        for k, v in inflections.items():
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
        w = re.sub(r'\(\?:-', r'(?:\S*', w)                 #     '\S*ing/[A-Z]+'
        w = re.sub(r'\(\?:(\S*)-/', r'(?:\\b\1\S*', w)
        w = re.sub(r'\/', '\\\/', w)
        p.append(w)

    p = '(%s)' % ''.join(p)
    for m in re.finditer(p, '%s ' % sentence, re.I):
        m = (m.strip() for m in m.groups() if m)
        m = map(Phrase, m)
        m = tuple(m)
        if len(m) > 1:
            yield m
        else:
            yield m[0]

# for m in \
#   chunk('ADJ', 
#     tag('A big, black cat.')):
#      print(u(m))

# for m, g1, g2 in \
#   chunk('DET? (NOUN) AUX? BE (-ry)', 
#     tag("The cats'll be hungry.")): 
#     print(u(g1), u(g2))

# for m, g1, g2 in \
#   chunk('DET? (NOUN) AUX? BE (-ry)', 
#     tag("The boss'll be angry!")):
#     print(u(g1), u(g2))

# for m, g1, g2, g3 in \
#   chunk('(NOUN|PRON) BE ADV? (ADJ) than (NOUN|PRON)', 
#     tag("Cats are more stubborn than dogs.")):
#     print(u(g1), u(g2), u(g3))

def constituents(sentence, language='en'):
    """ Yields an iterator of (Phrase, tag)-tuples from the given Sentence,
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
    s = u(sentence)
    s = re.sub(r'\s+', ' ', s)
    while s:
        for tag, p in P:
            try:
                m = next(chunk('^(%s)' % p, s))[0]; break
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

#---- SENTIMENT ANALYSIS --------------------------------------------------------------------------
# Sentiment analysis aims to determine the affective state of (subjective) text,
# for example whether a customer review is positive or negative about a product.

polarity = {
    'en': {
       u'😃'    : +1.0,
        'great' : +1.0,
        'good'  : +0.5,
        'nice'  : +0.5,
        'bad'   : -0.5,
        'awful' : -1.0,
       u'😠'    : -1.0,
    }
}

negation = {
    'en': set(('no', 'not', "n't")),
}

intensifiers = {
    'en': set(('really', 'very')),
}

for f in glob.glob(cd('*-pol.json')):
    polarity[f.split('-')[-2][-2:]] = json.load(open(f))

def sentiment(s, language='en'):
    """ Returns the polarity of the string as a float,
        from negative (-1.0) to positive (+1.0).
    """
    p = polarity.get(language, {})
    s = s.lower()
    s = s.split()
    a = []
    for i, w in enumerate(s):
        if w in p:
            n = p[w]
            if i > 0:
                if s[i-1] in negation.get(language, ()):     # not good
                    n = n * -1.0
                if s[i-1] in intensifiers.get(language, ()): # very bad
                    n = n * +1.5
            a.append(n)
    v = avg(a)
    v = max(v, -1.0)
    v = min(v, +1.0)
    return v

# s = 'This movie is very bad!'
# s = tokenize(s)
# s = sentiment(s)
# print(s)

def positive(s, language='en', threshold=0.1):
    # = 75% (Pang & Lee polarity dataset v1.0)
    return sentiment(s, language) >= threshold

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

def oauth(url, data={}, method='GET', key='', token='', secret=('','')):
    """ Returns (url, data), where data is updated with OAuth 1.0 authorization.
    """

    def nonce():
        return hashlib.md5(b('%s%s' % (time.time(), random.random()))).hexdigest()

    def timestamp():
        return int(time.time())

    def encode(s):
        return urlquote(b(s), safe='~')

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
    p = urlparse.urlsplit(url)
    q = urlparse.parse_qsl(p.query)
    q.extend((b(k), b(v)) for k, v in sorted(data.items()))
    q = urlencode(q, doseq=True)
    p = p.scheme, p.netloc, p.path, q, p.fragment
    s = urlparse.urlunsplit(p)
    s = s.lstrip('?')
    return s

# print(serialize('http://www.google.com', {'q': 'cats'})) # http://www.google.com?q=cats

#---- REQUESTS & STREAMS --------------------------------------------------------------------------
# The download(url) function returns the HTML (JSON, image data, ...) at the given url.
# If this fails it will raise NotFound (404), Forbidden (403) or TooManyRequests (420).

class Forbidden       (Exception): pass # 403
class NotFound        (Exception): pass # 404
class TooManyRequests (Exception): pass # 429
class Timeout         (Exception): pass

cookies = cookielib.CookieJar()

def request(url, data={}, headers={}, timeout=10):
    """ Returns a file-like object to the given URL.
    """

    if cookies is not None:
        f = urllib2.HTTPCookieProcessor(cookies)
        f = urllib2.build_opener(f)
    else:
        f = urllib2.build_opener()
    try:
        f = f.open(Request(url, urlencode(data) if data else None, headers), timeout=timeout)

    except URLError as e:
        status = getattr(e, 'code', None) # HTTPError
        if status == 401:
            raise Forbidden
        if status == 403:
            raise Forbidden
        if status == 404:
            raise NotFound
        if status == 420:
            raise TooManyRequests
        if status == 429:
            raise TooManyRequests
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
        s = request(url, data, headers, timeout).read()
        with open(k, 'wb') as f:
            f.write(s)

        time.sleep(max(0, download.ready - time.time())) # rate limiting
        download.ready = time.time() + delay

    with open(k, 'rb') as f:
        return f.read()

# print(u(download('http://textgain.com')))

class stream(list):

    def __init__(self, request, bytes=1024):
        """ Returns an iterator of read data for the given request().
        """
        self.request = request
        self.bytes = bytes

    def __iter__(self):
        b = '' # buffer
        while 1:
            try:
                b = b + self.request.read(self.bytes)
                b = b.split('\n')
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
#     if os.stat(f).st_ctime < t:
#         os.remove(f)

#---- SEARCH --------------------------------------------------------------------------------------
# The Bing Search API grants 5,000 free requests per month.
# The Google Search API grants 100 free requests per day.

keys = {
    'Bing'   : '4PYH6hSM/Asibu9Nn7MTE+lu/hViglqCl/rV20yBP5o',
    'Google' : 'AIzaSyBxe9jC4WLr-Rry_5OUMOZ7PCsEyWpiU48'
}

Result = collections.namedtuple('Result', ('url', 'text'))

def bing(q, page=1, language='en', delay=1, cached=False, key=None):
    """ Returns an iterator of (url, description)-tuples from Bing.
    """
    if 0 < page <= 100:
        r  = 'https://api.datamarket.azure.com/bing/search/Web'
        r += '?Query=\'%s'
        r += '+language:%s\''
        r += '&$skip=%i'
        r += '&$top=10'
        r += '&$format=json'
        r %= (
            urlquote(b(q)),
            urlquote(b(language)), 1 + 10 * (page - 1))
        k = base64.b64encode(b(':%s' % (key or keys['Bing'])))
        r = download(r, headers={'Authorization': b'Basic ' + k}, delay=delay, cached=cached)
        r = json.loads(u(r))

        for r in r['d']['results']:
            yield Result(
                r['Url'],
                r['Description']
            )

def google(q, page=1, language='en', delay=1, cached=False, key=None):
    """ Returns an iterator of (url, description)-tuples from Google.
    """
    if 0 < page <= 10:
        r  = 'https://www.googleapis.com/customsearch/v1'
        r += '?cx=000579440470800426354:_4qo2s0ijsi'
        r += '&key=%s' % (key or keys['Google'])
        r += '&q=%s'
        r += '&lr=lang_%s'
        r += '&start=%i'
        r += '&num=10'
        r += '&alt=json'
        r %= (
            urlquote(b(q)),
            urlquote(b(language)), 1 + 10 * (page - 1))
        r = download(r, delay=delay, cached=cached)
        r = json.loads(u(r))

        for r in r['items']:
            yield Result(
                r['link'],
                r['htmlSnippet']
            )

def search(q, engine='bing', page=1, language='en', delay=1, cached=False, key=None):
    """ Returns an iterator of (url, description)-tuples.
    """
    f = globals().get(engine, lambda *args: None)
    if key:
        return f(q, page, language, delay, cached, key)
    else:
        return f(q, page, language, delay, cached)

# for url, description in search('textgain'):
#     print(url)
#     print(description)
#     print('\n')

#---- TRANSLATE -----------------------------------------------------------------------------------

def translate(q, source='', target='en', delay=1, cached=False, key=None):
    """ Returns the translated string.
    """
    r  = 'https://www.googleapis.com/language/translate/v2?'
    r +=('&source=%s' % source) if source else ''
    r += '&target=%s' % target
    r += '&key=%s'    % key
    r += '&q=%s'      % urlquote(b(q[:1000]))
    r  = download(r, delay=delay, cached=cached)
    r  = json.loads(u(r))
    r  = r.get('data', {})
    r  = r.get('translations', [{}])[0]
    r  = r.get('translatedText', '')
    return r

setattr(google, 'search'   , search   )
setattr(google, 'translate', translate)

# print(google.translate('De zwarte kat zat op de mat.', source='nl', key='***'))

#---- TWITTER -------------------------------------------------------------------------------------
# Using u(), oauth(), request() and stream() we can define a Twitter class.
# Twitter.search(q) gives you tweets that contain the word q.
# Twitter.stream(q) gives you tweets that contain the word q, live as they are posted.
# Twitter.follow(q) gives you tweets posted by username q.
# Twitter.followers(q) gives you usernames of people that follow q.

# https://dev.twitter.com/docs/api/1.1
# https://dev.twitter.com/streaming/overview/request-parameters

keys['Twitter'] = OAuth(
    'zinzNx4FFyLDQkOaTnR9zYRq7',
    '2365345020-snrMR8jQ69WDZ0KbSGvF1b4O7kIyynJp9v3UySL', (
    'VFlV2M9mimg8bZTTct9qVuOVdWvak5MmCfghtdB6B8SOQvINbL',
    'MrsrcmKkyzWOTjoKVsPLVvCYRtMcDYaIx0NKIb6yhRIhv'
))

Tweet = collections.namedtuple('Tweet', ('id', 'text', 'date', 'language', 'author', 'photo', 'likes'))

class Twitter(object):

    def parse(self, v):
        def f(v):
            v = decode(v.get('extended_tweet', {}) \
                        .get('full_text',          # 240 characters (stream)
                       v.get('full_text',          # 240 characters (search)
                       v.get('text', ''))))        # 140 characters (< 2017)
            return v

        t = Tweet(
            u(v.get('id_str', '')),
            u(f(v)),
            u(v.get('created_at', '')),
            u(v.get('lang', '')).replace('und', ''),
            u(v.get('user', {}).get('screen_name', '')),
            u(v.get('user', {}).get('profile_image_url', '')),
          int(v.get('favorite_count', 0))
        )
        RT =  v.get('retweeted_status')
        if RT:
            # Replace truncated retweet (...) with full text.
            t = t._replace(text=u('RT @%s: %s' % (RT['user']['screen_name'], f(RT))))
        return t

    def stream(self, q, language='', timeout=60, delay=1, key=None):
        """ Returns an iterator of tweets (live).
        """
        k = key or keys['Twitter']
        r = 'https://stream.twitter.com/1.1/statuses/filter.json', {
            'language'   : language,
            'track'      : q
        }
        r = oauth(*r, key=k.key, token=k.token, secret=k.secret)
        r = serialize(*r)
        r = request(r, timeout=timeout)

        for v in stream(r):
            v = u(v)
            v = json.loads(v)
            v = self.parse(v)
            yield v
            time.sleep(delay)

    def search(self, q, language='', delay=5.5, cached=False, key=None):
        """ Returns an iterator of tweets.
        """
        id = ''
        for i in range(10):
            k = key or keys['Twitter']
            r = 'https://api.twitter.com/1.1/search/tweets.json', {
                'tweet_mode' : 'extended',
                'count'      : 100,
                'max_id'     : id,
                'lang'       : language,
                'q'          : q
            }
            r = oauth(*r, key=k.key, token=k.token, secret=k.secret)
            r = serialize(*r)
            r = download(r, delay=delay, cached=cached) # 180 requests / 15 minutes
            r = json.loads(u(r))
            r = r.get('statuses', [])

            for v in r:
                yield self.parse(v)
            if len(r) > 0:
                id = int(v['id_str']) - 1
            if len(r) < 100:
                raise StopIteration

    def follow(self, q, language='', delay=5.5, cached=False, key=None):
        """ Returns an iterator of tweets for the given username.
        """
        return self.search(u'from:' + q, language, delay, cached, key)

    def followers(self, q, delay=75, cached=False, key=None):
        """ Returns an iterator of followers for the given username.
        """
        id = -1
        while 1:
            k = key or keys['Twitter']
            r = 'https://api.twitter.com/1.1/followers/list.json', {
                'count'       : 200,
                'cursor'      : id,
                'screen_name' : q.lstrip('@')
            }
            r = oauth(*r, key=k.key, token=k.token, secret=k.secret)
            r = serialize(*r)
            r = download(r, delay=delay, cached=cached) # 15 requests / 15 minutes
            r = json.loads(u(r))

            for v in r.get('users', []):
                yield v.get('screen_name')
            try:
                id = r['next_cursor']
            except:
                raise StopIteration
            if id == 0:
                raise StopIteration

    def likes(self, q, delay=60, cached=False, headers={'User-Agent': 'Grasp.py'}):
        """ Returns an iterator of usernames that liked the tweet with the given id.
        """
        r = 'https://twitter.com/i/activity/favorited_popup?id=%s' % q
        r = download(r, headers=headers, delay=delay, cached=cached)
        r = json.loads(u(r))

        for v in set(re.findall(r'screen-name="(.*?)"', r.get('htmlUsers', ''))):
            yield v

    def __call__(self, *args, **kwargs):
        return self.search(*args, **kwargs)

twitter = Twitter()

# for tweet in twitter('cats', language='en'):
#     print(tweet.text)

# for tweet in twitter.stream('cats'):
#     print(tweet.text)

# for username in twitter.followers('textgain'):
#     print(username)

#---- WIKIPEDIA -----------------------------------------------------------------------------------

BIBLIOGRAPHY = set((
    'div'                ,
    'table'              , # infobox
    '#references'        , # references title
    '.reflist'           , # references
    '.reference'         , # [1]
    '.mw-editsection'    , # [edit]
    '.noprint'           , # [citation needed]
    'h2 < #see_also'     ,
    'h2 < #see_also ~ *' ,
    'h2 < #notes'        ,
    'h2 < #notes ~ *'    ,
))

def wikipedia(q='', language='en', delay=1, cached=True):
    """ Returns the HTML source of the given Wikipedia article (or '').
    """
    r  = 'https://%s.wikipedia.org/w/api.php' % language
    r += '?action=parse'
    r += '&format=json'
    r += '&redirects=1'
    r += '&page=%s' % urlquote(q)
    r  = download(r, delay=delay, cached=cached)
    r  = json.loads(u(r))

    try:
        return u'<h1>%s</h1>\n%s' % (
            u(r['parse']['title']),
            u(r['parse']['text']['*']))
    except KeyError:
        return u''

# 1. Parse HTML source

# src = cached(wikipedia, 'Arnold Schwarzenegger', language='en')
# dom = DOM(src) # see below

# 2. Parse article (full, plaintext):

# article = plaintext(dom, ignore=BIBLIOGRAPHY)
# print(article)

# 3. Parse summary (= 1st paragraph):

# summary = plaintext(dom('p')[0])
# print(summary)

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
            u(e.findtext('author'           , ''))
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

SELF_CLOSING = set(( # <img src="" />
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
))

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
            for n in self.parent.children[self.parent.children.index(self)+1:]:
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

    def find(self, tag='*', attributes={}, depth=1e10):
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
    r'([+<>~])?',                                               # combinator + < >
    r'(\w+|\*)?',                                               # tag
    r'((?:[.#][-\w]+)|(?:\[.*?\]))?',                           # attributes # . [=]
    r'(\:first-child|\:(?:nth-child|not|contains)\(.*?\))?',    # pseudo :
    r'$'
)))

CLASS = \
    r'(^|\s)%s(\s|$)'

def selector(element, s):
    """ Returns a list of nested elements that match the given CSS selector chain.
    """
    m = []
    s = s.strip()
    s = s.lower()                                               # case-insensitive
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'([,+<>~])\s', '\\1', s)
    s = s or '<>'

    for s in s.split(','):                                      # div, a
        e = [element]
        for s in s.split(' '):
            try:
                combinator, tag, a, pseudo = \
                    SELECTOR.search(s).groups('')
            except: 
                return []

            tag = tag or '*'                                    # *

            if not a:                                           # a
                a = {}
            elif a.startswith('#'):                             # a#id
                a = {   'id': re.compile(        a[1:], re.I)}
            elif a.startswith('.'):                             # a.class
                a = {'class': re.compile(CLASS % a[1:], re.I)}
            elif a.startswith('['):                             # a[href]
                a = a.strip('[]')
                a = a.replace('"', '')
                a = a.replace("'", '')

                k, op, v = (re.split(r'([\^\$\*]?=)', a, 1) + ['=', r'.*'])[:3]

                if op ==  '=':
                    a = {k: re.compile(r'^%s$' % v, re.I)}      # a[href="https://textgain.com"]
                if op == '^=':
                    a = {k: re.compile(r'^%s'  % v, re.I)}      # a[href^="https"]
                if op == '$=':
                    a = {k: re.compile(r'%s$'  % v, re.I)}      # a[href$=".com"]
                if op == '*=':
                    a = {k: re.compile(r'%s'   % v, re.I)}      # a[href*="textgain"]

            if combinator == '':
                e = (e.find(tag, a) for e in e)
                e = list(itertools.chain(*e))                   # div a
            if combinator == '>':
                e = (e.find(tag, a, 1) for e in e)
                e = list(itertools.chain(*e))                   # div > a
            if combinator == '<':
                e = [e for e in e if any(e.find(tag, a, 1))]    # div < a
            if combinator == '+':
                e = map(lambda e: e.next, e)
                e = [e for e in e if e and e.match(tag, a)]     # div + a
            if combinator == '~':
                e = map(lambda e: e.successors, e)
                e = (e for e in e for e in e if e.match(tag, a))
                e = list(unique(e))                             # div ~ a

            if pseudo.startswith(':first-child'):
                e = (e for e in e if not e.previous)
                e = list(unique(e))                             # div a:first-child
            if pseudo.startswith(':nth-child'):
                s = pseudo[10:].strip('()"\'')
                e = [e[int(s) - 1]]                             # div a:nth-child(2)
            if pseudo.startswith(':not'):
                s = pseudo[4:].strip('()"\'')
                e = [e for e in e if e not in element(s)]       # div:not(.main)
            if pseudo.startswith(':contains'):
                s = pseudo[9:].strip('()"\'')
                e = (e for e in e if s in e.html.lower())
                e = list(unique(e))                             # div:contains("hello")

        m.extend(e)
    return m

# dom = DOM(download('https://www.textgain.com'))
# 
# print(dom('#nav > h1 b')[0].html)
# print(dom('meta[name="description"]')[0].content)
# print(dom('a[href^="https"]:first-child'))
# print(dom('a[href^="https"]:contains("love")'))

#---- PLAINTEXT -----------------------------------------------------------------------------------
# The plaintext() function traverses a DOM HTML element, strips all tags while keeping Text data.

BLOCK = set((
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
))

PLAIN = {
     'li' : lambda s: '* %s\n' % re.sub(r'\n\s+', '\n&nbsp;&nbsp;', s),
     'h1' : lambda s: s + '\n' + "-" * len(s),
     'h2' : lambda s: s + '\n' + "-" * len(s),
     'br' : lambda s: s + '\n'  ,
     'tr' : lambda s: s + '\n\n',
     'th' : lambda s: s + '\n'  ,
     'td' : lambda s: s + '\n'  ,
}

def plaintext(element, keep={}, ignore=set(('head', 'script', 'style', 'form')), format=PLAIN):
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
                s += re.sub(r'\s+', ' ', unescape(n.data))
            if isinstance(n, Element):
                if n in ignore:
                    continue
                if n.tag in BLOCK:
                    s += '\n\n'
                if n.tag in keep:
                    a  = ' '.join(['%s=%s' % (k, quote(n[k])) for k in keep[n.tag] if n[k] != None])
                    a  = ' ' + a if a else ''
                    s += '<%s%s>%s</%s>' % (n.tag, a, r(n), n.tag)
                else:
                    s += format.get(n.tag, lambda s: s)(r(n))
                if n.tag in BLOCK:
                    s += '\n\n'
        return s.strip()

    s = r(element)
    s = re.sub(r'(\s) +'       , '\\1'  , s) # no left indent
    s = re.sub(r'&nbsp;'       , ' '    , s) # exdent bullets
    s = re.sub(r'\n +\*'       , '\n*'  , s) # dedent bullets
    s = re.sub(r'\n\* ?(?=\n)' , ''     , s) # no empty lists
    s = re.sub(r'\n\n+'        , '\n\n' , s) # no empty lines
    s = s.strip()
    return s

# dom = DOM(download('https://www.textgain.com'))
# txt = plaintext(dom, keep={'a': ['href']})
# 
# print(txt)

def encode(s):
    """ Returns a string with encoded entities.
    """
    s = s.replace('&' , '&amp;' )
    s = s.replace('<' , '&lt;'  )
    s = s.replace('>' , '&gt;'  )
    s = s.replace('"' , '&quot;')
    s = s.replace("'" , '&apos;')
   #s = s.replace('\n', '&#10;' )
   #s = s.replace('\r', '&#13;' )
    return s

def decode(s):
    """ Returns a string with decoded entities.
    """
    s = s.replace('&amp;'  , '&')
    s = s.replace('&lt;'   , '<')
    s = s.replace('&gt;'   , '>')
    s = s.replace('&quot;' , '"')
    s = s.replace('&apos;' , "'")
    s = s.replace('&nbsp;' , ' ')
   #s = s.replace('&#10;'  , '\n')
   #s = s.replace('&#13;'  , '\r')
    s = re.sub(r'https?://.*?(?=\s|$)', \
        lambda m: urldecode(m.group()), s) # '%3A' => ':' (in URL)
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
    # Date.minute
    # Date.second

    @property
    def week(self):
        return self.isocalendar()[1]

    @property
    def weekday(self):
        return self.isocalendar()[2]

    @property
    def timestamp(self):
        return int(time.mktime(self.timetuple()))

    def format(self, s):
        return u(self.strftime(s))

    def __str__(self):
        return self.strftime('%Y-%m-%d %H:%M:%S')

    def __add__(self, i):
        return date(datetime.datetime.__add__(self, datetime.timedelta(seconds=i)))

    def __sub__(self, i):
        return date(datetime.datetime.__sub__(self, datetime.timedelta(seconds=i)))

    def __repr__(self):
        return "Date(%s)" % repr(str(self))

def date(*v, **format):
    """ Returns a Date from the given timestamp or date string.
    """
    format = format.get('format', '%Y-%m-%d %H:%M:%S')

    if len(v) > 1:
        return Date(*v) # (year, month, day, ...)
    if len(v) < 1:
        return Date.now()
    else:
        v = v[0]
    if isinstance(v, (int, float)):
        return Date.fromtimestamp(v)
    if isinstance(v, datetime.datetime):
        return Date.fromtimestamp(time.mktime(v.timetuple()))
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

# print(date('Dec 31 1999', format='%b %d %Y'))
# print(date(1999, 12, 31))

##### WWW #########################################################################################

#---- APP -----------------------------------------------------------------------------------------

# { 404: ('Not Found', 'Nothing matches the given URI')}
STATUS = BaseHTTPServer.BaseHTTPRequestHandler.responses
STATUS[429] = ('Too Many Requests', '')

SECOND, MINUTE, HOUR, DAY = 1, 1*60, 1*60*60, 1*60*60*24

# A pool of recycled threads is more efficient than a
# thread / request (see SocketServer.ThreadingMixIn).
class ThreadPoolMixIn(SocketServer.ThreadingMixIn):

    def __init__(self, threads=10):
        self.pool = multiprocessing.pool.ThreadPool(threads)

    def process_request(self, *args):
        self.pool.apply_async(self.process_request_thread, args)

class RouteError(Exception):
    pass

class Router(dict):

    def __setitem__(self, path, f):
        """ Defines the handler function for the given path.
        """
        return dict.__setitem__(self, path.strip('/'), f)

    def __getitem__(self, path):
        """ Returns the handler function for the given path.
        """
        return dict.__getitem__(self, path.strip('/'))

    def __call__(self, path, query):
        """ Returns the value of the handler for the given path,
            or the parent path if no handler is found.
        """
        path = path.strip('/')
        path = path.split('/')
        for i in reversed(range(len(path) + 1)):
            try:
                f = self['/'.join(path[:i])]
            except:
                continue
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

WSGIServer = wsgiref.simple_server.WSGIServer

class App(ThreadPoolMixIn, WSGIServer):

    def __init__(self, host='127.0.0.1', port=8080, threads=10):
        """ A multi-threaded web app served by a WSGI-server, that starts with App.run().
        """
        WSGIServer.__init__(self, (host, port), wsgiref.simple_server.WSGIRequestHandler)
        ThreadPoolMixIn.__init__(self, threads)
        self.set_app(self.__call__)
        self.rate     = {}
        self.router   = Router()
        self.request  = HTTPRequest()
        self.response = HTTPResponse()

    def route(self, path, rate=None, key=lambda request: request.ip):
        """ The @app.route(path) decorator defines the handler for the given path.
            The handler(*path, **query) returns a str or dict for the given path.
            With rate=(n, t), the IP-address is granted n requests per t seconds,
            before raising a 429 Too Many Requests error.
        """
        # http://127.0.0.1:8080/api/tag?q=Hello+world!&language=en
        # app = App()
        # @app.route('/api/tag')
        # def api_tag(q='', language='en', rate=(100, HOUR)):
        #     return '%s' % tag(q)
        def decorator(f):
            def wrapper(*args, **kwargs):
                if rate:
                    t = time.time()                    # now
                    i, d = self.rate.get(key, (0, t))  # used, since
                    if rate[1] < t - d:                # now - since > interval?
                        n = 0
                    if rate[0] < i + 1:                # used > limit?
                        raise HTTPError(429)
                    self.rate[key] = (i + 1, t)
                return f(*args, **kwargs)
            self.router[path] = wrapper
            return wrapper
        return decorator

    def run(self, debug=True):
        """ Starts the server.
        """
        print('Starting server at %s:%s... press ctrl-c to stop.' % self.server_address)
        self.debug = debug
        self.serve_forever()

    def __call__(self, env, start_response):

        # Parse HTTP headers.
        # 'HTTP_USER_AGENT' => 'User-Agent'
        def headers(env):
            for k, v in env.items():
                if k.startswith('HTTP_'):
                    k = k[5:]
                    k = k.replace('_', '-')
                    k = k.title()
                    yield u(k), u(v)

        # Parse HTTP GET and POST data.
        # '?page=1' => (('page', '1'),)
        def query(env):
            GET, POST = (
                env['QUERY_STRING'],
                env['wsgi.input'].read(int(env.get('CONTENT_LENGTH') or 0))
            )
            for k, v in urlparse.parse_qs(GET , True).items():
                yield u(k), u(v[-1])
            for k, v in urlparse.parse_qs(POST, True).items():
                yield u(k), u(v[-1])

        # Set App.request (thread-safe).
        r = self.request
        r.__dict__.update({
            'app'     : self,
            'ip'      : env['REMOTE_ADDR'],
            'method'  : env['REQUEST_METHOD'],
            'path'    : env['PATH_INFO'],
            'query'   : dict(query(env)),
            'headers' : dict(headers(env)),
        })

        # Set App.response (thread-safe).
        r = self.response
        r.__dict__.update({
            'code'    : 200,
            'headers' : {
                'content-type': 'text/html; charset=utf-8', 
                'access-control-allow-origin': '*' # CORS
            }
        })

        try:
            v = self.router(
                self.request.path, 
                self.request.query
            )
        except Exception as e:
            if   isinstance(e, HTTPError):
                r.code = e.code
            elif isinstance(e, RouteError):
                r.code = 404
            else:
                r.code = 500
            # Error page (with traceback if debug=True):
            v  = '<h1>%s</h1><p>%s</p>' % STATUS[r.code]
            v += '<pre>%s</pre>' % traceback.format_exc() \
                    if self.debug else ''

        if isinstance(v, dict): # dict => JSON-string
            r.headers['content-type'] = 'application/json'
            v = json.dumps(v)
        if hasattr(v, '__str__'):
            v = v.__str__()
        if v is None:
            v = ''

        # https://www.python.org/dev/peps/pep-0333/#the-start-response-callable
        start_response('%s %s' % (r.code, STATUS[r.code]), list(r.headers.items()))
        return [b(v)]

try:
    app = application = App(threads=10)
except socket.error:
    pass # 'Address already in use'

# http://127.0.0.1:8080/products?page=1
# @app.route('/')
# def index(*path, **query):
#     #raise HTTPError(500)
#     return 'Hello world! %s %s' % (repr(path), repr(query))

# http://127.0.0.1:8080/api/tag?q=Hello+world!&language=en
# @app.route('/api/tag', rate=(10, MINUTE))
# def api_tag(q='', language='en'):
#     return tag(q, language)

# app.run()

##### NET #########################################################################################

#---- GRAPH ---------------------------------------------------------------------------------------
# A graph is a set of nodes and edges (i.e., connections between nodes).
# A graph is stored as an adjacency matrix {node1: {node2: edge weight}}.
# It can then be analysed to find the shortest paths (e.g., navigation),
# clusters (e.g., communities in social networks), the strongest nodes
# (e.g., search engines), and so on.

# Edges can have a weight, which is the cost or length of the path.
# Edges can have a type, for example 'is-a' or 'is-part-of'.

def dfs(g, n, f=lambda n: True, v=set()):
    """ Depth-first search.
        Calls f(n) on the given node, its adjacent nodes if True, and so on.
    """
    v.add(n) # visited?
    if f(n) != False:
        for n in g.get(n, {}).keys():
            if n not in v:
                dfs(g, n, f, v)

def bfs(g, n, f=lambda n: True, v=set()):
    """ Breadth-first search (spreading activation).
        Calls f(n) on the given node, its adjacent nodes if True, and so on.
    """
    q = collections.deque([n])
    while q:
        n = q.popleft()
        if n not in v and f(n) != False:
            q.extend(g.get(n, {}).keys())
            v.add(n)

# def visit(n):
#     print(n)
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
        d, n, p = heappop(q)
        if n not in v:
            v.add(n)
            p += (n,)
            if n2 == None and n1 != n:
                yield p
            if n2 != None and n2 == n:
                yield p
                raise StopIteration
            for n, w in g.get(n, {}).items(): # {n1: {n2: cost}}
                if n not in v:
                    heappush(q, (d + w, n, p))

def shortest_path(g, n1, n2):
    """ Returns the shortest path from n1 to n2.
    """
    try:
        return next(shortest_paths(g, n1, n2))
    except StopIteration:
        return None

# g = {
#     'a': {'b': 1, 'x': 1},  # A -> B -> C -> D
#     'b': {'c': 1},          #   –> X ––––––>
#     'c': {'d': 1},
#     'x': {'d': 1}
# }
# print(shortest_path(g, 'a', 'd'))

def betweenness(g, k=1000):
    """ Returns a dict of node id's and their centrality score (0.0-1.0),
        which is the amount of shortest paths that pass through a node.
    """
    n = set().union(g, *g.values()) # all nodes
    w = collections.Counter(n)
    if k:
        n = list(shuffled(n))[:k]
    for n in n:
        for p in shortest_paths(g, n):
            for n in p[1:-1]:
                w[n] += 1
    # Normalize 0.0-1.0:
    m = max(w.values())
    m = float(m or 1)
    w = {n: w / m for n, w in w.items()}
    return w

def pagerank(g, iterations=100, damping=0.85, epsilon=0.00001):
    """ Returns a dict of node id's and their centrality score (0.0-1.0),
        which is the amount of indirect incoming links to a node.
    """
    n = set().union(g, *g.values()) # all nodes
    v = dict.fromkeys(n, 1.0 / len(n))
    for i in range(iterations):                            #       A -> B -> C
        p = v.copy() # prior pagerank                      #      0.3  0.3  0.3
        for n1 in v:                                       # i=0  0.3  0.6  0.6
            for n2, w in g.get(n1, {}).items():            # i=1  0.3  0.9  1.2
                v[n2] += damping * w * p[n1] / len(g[n1])  # i=2  0.3  1.2  2.1
            v[n1] += 1 - damping                           # ...
        # Normalize:
        d = sum(w ** 2 for w in v.values()) ** 0.5 or 1
        v = {n: w / d for n, w in v.items()}
        # Converged?
        e = sum(abs(v[n] - p[n]) for n in v)
        if e < epsilon * len(n):
            break
    return v

def cliques(g):
    """ Returns an iterator of maximal cliques,
        where each clique is a set of node id's
        that are all connected to each other.
    """

    # Bron-Kerbosch's backtracking algorithm.
    def search(r, p, x):
        if p or x:
            u = p | x # pivot
            u = u.pop()
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
    n = set().union(g, *g.values()) # all nodes
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

    def __init__(self, directed=False):
        self._directed = directed

    @property
    def directed(self):
        return self._directed

    @property
    def density(self):
        n = len(list(self.nodes))
        e = len(list(self.edges))
        return float(self.directed + 1) * e / n / (n - 1) # 0.0-1.0

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
        g = self.__class__(directed=self._directed)
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
        if n2 != None and not self._directed:
            self[n2][n1] = e

    def pop(self, n1, n2=None, default=None):
        if n2 == None:
            for n in self:
                self[n].pop(n1, None)       # n1 <- ...
            v = dict.pop(self, n1, default) # n1 -> ...
        if n2 != None:
            v = self.get(n1, {}).pop(n2, default)
        if n2 != None and not self._directed:
            v = self.get(n2, {}).pop(n1, default)
        return v

    def sub(self, nodes=[]):
        """ Returns a graph with the given nodes, and connecting edges.
        """
        g = self.__class__(directed=self._directed)
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
        g = self.__class__(directed=self._directed)
        g.add(n)
        for i in range(depth):
            for e in [e for e in self.edges if not e in g and e.node1 in g or e.node2 in g]:
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
        s  = '<?xml version="1.0" encoding="utf-8"?>'
        s += '\n<graphml>'
        s += '\n<key for="edge" id="weight" attr.name="weight" attr.type="float"/>'
        s += '\n<key for="edge" id="type" attr.name="type" attr.type="string"/>'
        s += '\n<graph edgedefault="%sdirected">' % ('un', '')[self._directed]
        for n in self.nodes:
            s += '\n<node id="%s" />' % escape(n)
        for e in self.edges:
            s += '\n<edge source="%s" target="%s">' % (escape(e.node1), escape(e.node2))
            s += '\n\t<data key="weight">%s</data>' % (e.weight)
            s += '\n\t<data key="type">%s</data>'   % (e.type or '')
            s += '\n</edge>'
        s += '\n</graph>'
        s += '\n</graphml>'
        f = codecs.open(path, 'w', encoding='utf-8')
        f.write(s)
        f.close()

    @classmethod
    def load(cls, path):
        t = ElementTree
        t = t.parse(path)
        t = t.find('graph')
        d = t.get('edgedefault') == 'directed'
        g = cls(directed=d)
        for n in t.iter('node'): g.add(unescape(n.get('id')))
        for e in t.iter('edge'): g.add(
            unescape(e.get('source')), 
            unescape(e.get('target')), 
            e.findtext('*[@key="weight"]', 1.0),
            e.findtext('*[@key="type"]') or None
        )
        return g

def union(g1, g2):
    # g1 | g2
    g = g1.__class__(directed=g1.directed)
    g.update(g1)
    g.update(g2)
    return g

def intersection(g1, g2):
    # g1 & g2
    g = g1.__class__(directed=g1.directed)
    for n in g1.nodes:
        if n in g2 is True:
            g.add(n)
    for e in g1.edges:
        if e in g2 is True:
            g.add(*e)

def difference(g1, g2):
    # g1 - g2
    g = g1.__class__(directed=g1.directed)
    for n in g1.nodes:
        if n in g2 is False:
            g.add(n)
    for e in g1.edges:
        if e in g2 is False:
            g.add(*e)

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
        '\tvar adjacency = %s;' % json.dumps(a),
        '',
        '\tvar canvas;',
        '\tcanvas = document.getElementById(%(id)s);',
        '\tcanvas.graph = new Graph(adjacency);',
        '\tcanvas.graph.animate(canvas, %(n)s, {',
        '\t\tdirected    : %s,' % f('directed', False),
        '\t\tfont        : %s,' % f('font', '10px sans-serif'),
        '\t\tfill        : %s,' % f('fill', '#fff'),
        '\t\tstroke      : %s,' % f('stroke', '#000'),
        '\t\tstrokewidth : %s,' % f('strokewidth', 0.5),
        '\t\tradius      : %s,' % f('radius', 4.0),
        '\t\tf1          : %s,' % f('f1', 10.0),
        '\t\tf2          : %s,' % f('f2', 0.5),
        '\t\tm           : %s'  % f('m', 0.25),
        '\t});',
        '</script>'
    ))
    k = {}
    k.update({'src': 'graph.js', 'id': 'g', 'width': 640, 'height': 480, 'n': 1000})
    k.update(kwargs)
    k = {k: json.dumps(v) for k, v in k.items()}
    return s % k

# g = Graph()
# n = range(200)
# for i in range(200):
#     g.add(
#         n1=random.choice(n), 
#         n2=random.choice(n))
# 
# f = open('test.html', 'w')
# f.write(visualize(g, n=1000, directed=True))
# f.close()
