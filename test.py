# encoding: utf-8

from grasp import *

#---- ETC -----------------------------------------------------------------------------------------

a = [1, 2, 3, 4, 4]

assert list( first(2, iter(a)                  )) == [1, 2]
assert list(   sliced(iter(a)            , 0, 2)) == [1, 2]
assert list(   sliced(iter(a)         , 0, 3, 2)) == [1, 3]
assert list(   unique(iter(a)                  )) == [1, 2, 3, 4]
assert list(   chunks(iter(a)               , 2)) == [(1, 2), (3, 4)]
assert list(    nwise(iter(a)               , 2)) == [(1, 2), (2, 3), (3, 4), (4, 4)]

#---- ML ------------------------------------------------------------------------------------------

v1 = {'x': 1, 'y': 2}
v2 = {'x': 3, 'y': 4}
v3 = {'x': 5, 'y': 6}
v4 = {'x': 7, 'y': 0}

assert round( distance(v1, v2)               , 2) == 2.83
assert round(      dot(v1, v4)               , 2) == 7.00
assert round(     norm(v1)                   , 2) == 2.24
assert round(      cos(v1, v2)               , 2) == 0.02
assert round(     diff(v1, v2)               , 2) == 0.00
assert round(      knn(v1, (v2, v3))[0][0]   , 2) == 0.98
assert             knn(v1, (v2, v3))[0][1]        ==  v2
assert  list(   reduce(v1, 'y').keys()          ) == ['x']
assert          sparse(v4)                        == {'x': 7}
assert    binary(scale(v1, 0, 1))                 == {'x': 0, 'y': 1}
assert              tf(v4)                        == {'x': 1, 'y': 0}
assert       centroid((v1, v2))                   == {'x': 2, 'y': 3}
assert       features((v1, v2))                   == set(('x', 'y'))
assert next(iter(tfidf((v1, v2))))['x']           == 0.25
assert       majority([1, 1, 2])                  == 1

#---- ML ------------------------------------------------------------------------------------------

assert list(  chngrams('cats', 2))                == ['ca', 'at', 'ts']
assert list(    ngrams(('cats', '&', 'dogs'), 2)) == [('cats', '&'), ('&', 'dogs')]
assert list(    ngrams('cats & dogs', 2))         == [('cats', '&'), ('&', 'dogs')]
assert list( skipgrams('cats & dogs', 1))         == [('cats', ('&',)), ('&', ('cats', 'dogs')), 
                                                      ('dogs', ('&',))]

#---- NLP -----------------------------------------------------------------------------------------

assert similarity('cat', 'can', f=chngrams, n=2 ) == 0.5
assert similarity('cat', 'can', f=chngrams, n=1 ) == 2/3.

assert readability('the cat sat on the mat'     ) == 1.0
assert readability('felis silvestris catus'     ) == 0.0

assert collapse(u'a    b  c'                    ) == u'a b c'
assert cap     (u'cat. cat.'                    ) == u'Cat. Cat.'
assert sep     (u"'a's b, c (d).'"              ) == u"'a 's b , c ( d ) . '"
assert encode  (u'<a> & <b>'                    ) == u'&lt;a&gt; &amp; &lt;b&gt;'
assert decode  (u'&lt;a&gt; &amp; &lt;b&gt;'    ) == u'<a> & <b>'
assert decode  (u'http://google.com?q=%22x%22'  ) == u'http://google.com?q="x"'
assert decode  (u'decode("%22") = "%22"'        ) == u'decode("%22") = "%22"'
assert detag   (u'<a>b</a>\nc'                  ) == u'b\nc'
assert detag   (u'<a>a</a>&<b>b</b>'            ) == u'a&b'
assert destress(u'pâté'                         ) == u'pate'
assert deflood (u'Woooooow!!!!!!'         , n=3 ) == u'Wooow!!!'
assert decamel (u'HTTPError404NotFound'         ) == u'http_error_404_not_found'
assert sg      (u'cats'                         ) == u'cat'
assert sg      (u'mice'                         ) == u'mouse'
assert sg      (u'pussies'                      ) == u'pussy'
assert sg      (u'cheeses'                      ) == u'cheese'

assert top(lang(u'The cat is snoring'       ))[0] == 'en'
assert top(lang(u'De kat is aan het snurken'))[0] == 'nl'

assert top(loc( u'Cats conquered Paris'))[0].city == 'Paris'

assert hilite  (u'a b', {'a': 1.0}              ) == '<mark style="background: #ff0f;">a</mark> b'

assert scan(    u'Here, kitty kitty!'           ) == [4, 17]
assert scan(    u'Here, kitty!'                 ) == [4, 11]

#---- NLP -----------------------------------------------------------------------------------------

assert tokenize(u'(hello) (:'                   ) == u'( hello ) (:'
assert tokenize(u'References [123] (...) :-)'   ) == u'References [123] (...) :-)'
assert tokenize(u'Cats &amp; dogs &#38; etc.'   ) == u'Cats &amp; dogs &#38; etc.'
assert tokenize(u"I'll eat pizza w/ tuna :-)"   ) == u"I 'll eat pizza w/ tuna :-)"
assert tokenize(u'Google (http://google.com)'   ) == u'Google ( http://google.com )'
assert tokenize(u'Also see: www.google.com.'    ) == u'Also see : www.google.com .'
assert tokenize(u'e.g., google.de, google.be'   ) == u'e.g. , google.de , google.be'
assert tokenize(u'One! 😍 Two :) :) Three'      ) == u'One ! 😍\nTwo :) :)\nThree'
assert tokenize(u'Aha!, I see:) Super'          ) == u'Aha ! , I see :)\nSuper'
assert tokenize(u"U.K.'s J. R. R. Tolkien"      ) == u"U.K. 's J. R. R. Tolkien"
assert tokenize(u'your.name@gmail.com!'         ) == u'your.name@gmail.com !'
assert tokenize(u'http://google.com?p=1'        ) == u'http://google.com?p=1'
assert tokenize(u'"Citation." Next p.5'         ) == u'" Citation . "\nNext p.5'
assert tokenize(u'"Oh! Nice!" he said'          ) == u'" Oh ! Nice ! " he said'
assert tokenize(u'"Oh!" Nice.'                  ) == u'" Oh ! "\nNice .'
assert tokenize(u'Oh... Nice.'                  ) == u'Oh ...\nNice .'
assert tokenize(u'Oh! #wow Nice.'               ) == u'Oh ! #wow\nNice .'
assert tokenize(u'Hello.Hello! 20:10'           ) == u'Hello .\nHello !\n20:10'
assert tokenize(u'pre-\ndetermined'             ) == u'predetermined'

assert Tagged('//PUNC')                           == [('/', 'PUNC')]

s1 = u"Here/PRON 's/VERB my/PRON new/ADJ cool/ADJ cat/NOUN café/NOUN !/PUNC :-D/:)"
s2 = u'1\/2/NUM'

assert list(map(u, chunk(r'^? ?'          , s1))) == [u"Here/PRON 's/VERB"]
assert list(map(u, chunk(r'^-'            , s1))) == [u'Here/PRON']
assert list(map(u, chunk(r'^PRON'         , s1))) == [u'Here/PRON']
assert list(map(u, chunk(r'PRON'          , s1))) == [u'Here/PRON', u'my/PRON']
assert list(map(u, chunk(r'BE'            , s1))) == [u"'s/VERB"]
assert list(map(u, chunk(r'VERB'          , s1))) == [u"'s/VERB"]
assert list(map(u, chunk(r'AD[JV]'        , s1))) == [u'new/ADJ', u'cool/ADJ']
assert list(map(u, chunk(r'ADJ'           , s1))) == [u'new/ADJ', u'cool/ADJ']
assert list(map(u, chunk(r'ADJ+'          , s1))) == [u'new/ADJ cool/ADJ']
assert list(map(u, chunk(r'ADJ+ NOUN'     , s1))) == [u'new/ADJ cool/ADJ cat/NOUN']
assert list(map(u, chunk(r'ADJ|NOUN+'     , s1))) == [u'new/ADJ cool/ADJ cat/NOUN café/NOUN']
assert list(map(u, chunk(u'ADJ ca-'       , s1))) == [u'cool/ADJ cat/NOUN']
assert list(map(u, chunk(u'ca-'           , s1))) == [u'cat/NOUN', u'café/NOUN']
assert list(map(u, chunk(u'Ca-'           , s1))) == [u'cat/NOUN', u'café/NOUN']
assert list(map(u, chunk(u'-é'            , s1))) == [u'café/NOUN']
assert list(map(u, chunk(u'-é/NOUN'       , s1))) == [u'café/NOUN']
assert list(map(u, chunk(u'-é/noun'       , s1))) == []
assert list(map(u, chunk(u'-é/VERB'       , s1))) == []
assert list(map(u, chunk(r'NOUN ? ?$'     , s1))) == [u'café/NOUN !/PUNC :-D/:)']
assert list(map(u, chunk(r'PUNC ?$'       , s1))) == [u'!/PUNC :-D/:)']
assert list(map(u, chunk(r':\) ?$'        , s1))) == [u':-D/:)']
assert list(map(u, chunk(r':\)'           , s1))) == [u':-D/:)']
assert list(map(u, chunk(r'1/2'           , s2))) == []
assert list(map(u, chunk(r'1\/2'          , s2))) == [u'1\\/2/NUM']
assert list(map(u, chunk(r'1\/2/NUM'      , s2))) == [u'1\\/2/NUM']

assert u(list( constituents(s1))[0][0])           == u'Here/PRON'
assert u(list( constituents(s1))[1][0])           == u"'s/VERB"
assert u(list( constituents(s1))[2][0])           == u'my/PRON new/ADJ cool/ADJ cat/NOUN café/NOUN'
assert u(list( constituents(s1))[3][0])           == u'!/PUNC'
assert   list( constituents(s1))[0][1]            == u'NP'
assert   list( constituents(s1))[1][1]            == u'VP'
assert   list( constituents(s1))[2][1]            == u'NP'
assert   list( constituents(s1))[3][1]            == u''

#---- NLP -----------------------------------------------------------------------------------------

t1 = trie({
    'abc ' : 1,
    'abc*' : 2,
    'x .'  : 3,
    '. y'  : 4,
    'x y'  : 5,
    'xyz'  : 6,
    'ij.'  : 7,
    '*n'   : 8,
    'n?'   : 9,
    'q?q'  : 10,
    '  q'  : 11,
})

t2 = trie({
    'a'    : 1,
    'a b'  : 2,
    'a b c': 3,
    'a b *': 4,
    'a * *': 5,
    '* * c': 6,
    'x* *' : 7,
    'y. *' : 8,
    'y ..' : 9,
    'y .?' : 10,
    'z ?.' : 11,
    ''     : 12,
})

assert len(list(t1.search('abc d'             ))) == 0 # (debatable)
assert len(list(t1.search('abc '   , sep=None ))) == 1
assert len(list(t1.search('abcd'   , etc='*'  ))) == 1
assert len(list(t1.search('abc abc', etc='*'  ))) == 2
assert len(list(t1.search('x y'    , etc='*.' ))) == 3
assert len(list(t1.search('x y'               ))) == 1
assert len(list(t1.search('xyz'               ))) == 1
assert len(list(t1.search('xyzz'   , sep=None ))) == 1
assert len(list(t1.search('xyzz'              ))) == 0
assert len(list(t1.search('ijkk'   , etc='*.' ))) == 0
assert len(list(t1.search('ijk'    , etc='*.' ))) == 1
assert len(list(t1.search('nnn'    , etc='*.' ))) == 1
assert len(list(t1.search('nn'     , etc='*.?'))) == 2
assert len(list(t1.search('nn n'   , etc='*.' ))) == 2
assert len(list(t1.search('ijn nm' , etc='*.?'))) == 3
assert len(list(t1.search('nn n'   , etc='*.?'))) == 4
assert len(list(t1.search('abcn mn', etc='*.?'))) == 3
assert len(list(t1.search('abcn'   , etc='*.?'))) == 2
assert len(list(t1.search('abc*'   , sep=None ))) == 1
assert len(list(t1.search('abcd'   , sep=None ))) == 0
assert len(list(t1.search('aijn', 0, etc='*.' ))) == 2
assert len(list(t1.search('aijn', 0, etc='*.?'))) == 3
assert len(list(t1.search('nm'  , 0, etc='*.?'))) == 3 # (debatable)
assert len(list(t1.search('qx'     , etc='*.?'))) == 0
assert len(list(t1.search('qxq'    , etc='*.?'))) == 1
assert len(list(t1.search('qyq'    , etc='*.?'))) == 1
assert len(list(t1.search('  q'               ))) == 1
assert len(list(t1.search('  qq', 0, etc='*.?'))) == 2
assert len(list(t1.search('qq'     , etc='*.?'))) == 1

assert len(list(t2.search('a a a'             ))) == 3
assert len(list(t2.search('a b c'             ))) == 3
assert len(list(t2.search('a b _'  , etc='*.?'))) == 4
assert len(list(t2.search('a b c'  , etc='*.?'))) == 6
assert len(list(t2.search('b b c'  , etc='*.?'))) == 1
assert len(list(t2.search('a c _'  , etc='*.?'))) == 2
assert len(list(t2.search('x y'    , etc='*.?'))) == 1
assert len(list(t2.search('xx y'   , etc='*.?'))) == 1
assert len(list(t2.search('yx x'   , etc='*.?'))) == 1
assert len(list(t2.search('y x'    , etc='*.?'))) == 1
assert len(list(t2.search('y xx'   , etc='*.?'))) == 2
assert len(list(t2.search('y xxx'  , etc='*.?'))) == 0
assert len(list(t2.search('z x'    , etc='*.?'))) == 0 # bug! (#11)
assert len(list(t2.search('zx x'   , etc='*.?'))) == 0

assert subs([0, 3, 'cats', 1], [0, 2, 'cat', 1 ]) == True

#---- WWW -----------------------------------------------------------------------------------------

e1 = DOM('<div id="main"><div class="story"><p>1</p><p>2</p></div</div>')
e2 = DOM('<div><a href="http://www.site.com">x</a></div>')

assert list(map(u, e1('#main'                 ))) == [u(e1)]
assert list(map(u, e1('*#main'                ))) == [u(e1)]
assert list(map(u, e1('div#main'              ))) == [u(e1)]
assert list(map(u, e1('.story'                ))) == [u'<div class="story"><p>1</p><p>2</p></div>']
assert list(map(u, e1('div div'               ))) == [u'<div class="story"><p>1</p><p>2</p></div>']
assert list(map(u, e1('div < p'               ))) == [u'<div class="story"><p>1</p><p>2</p></div>']
assert list(map(u, e1('div > p'               ))) == [u'<p>1</p>', u'<p>2</p>']
assert list(map(u, e1('div div p'             ))) == [u'<p>1</p>', u'<p>2</p>']
assert list(map(u, e1('div p:first-child'     ))) == [u'<p>1</p>']
assert list(map(u, e1('div p:nth-child(1)'    ))) == [u'<p>1</p>']
assert list(map(u, e1('div p:nth-child(2)'    ))) == [u'<p>2</p>']
assert list(map(u, e1('p:not(":first-child")' ))) == [u'<p>2</p>']
assert list(map(u, e1('div p:contains("2")'   ))) == [u'<p>2</p>']
assert list(map(u, e1('p + p'                 ))) == [u'<p>2</p>']
assert list(map(u, e1('p ~ p'                 ))) == [u'<p>2</p>']
assert list(map(u, e2('*[href]'               ))) == [u'<a href="http://www.site.com">x</a>']
assert list(map(u, e2('a[href^="http://"]'    ))) == [u'<a href="http://www.site.com">x</a>']
assert list(map(u, e2('a[href$=".com"]'       ))) == [u'<a href="http://www.site.com">x</a>']
assert list(map(u, e2('a[href*="site"]'       ))) == [u'<a href="http://www.site.com">x</a>']

assert plaintext(e2                             ) == u'x'
assert plaintext(e2, keep={'a': []}             ) == u'<a>x</a>'
assert plaintext(e2, keep={'a': ['href']}       ) == u'<a href="http://www.site.com">x</a>'
assert plaintext(e2, ignore='a'                 ) == u''
assert plaintext(u'<a>b</a>\nc'                 ) == u'b c'
assert plaintext('<a>b </a>c'                   ) == u'b c'
assert plaintext('<p>b </p>c'                   ) == u'b\n\nc'

#---- ETC -----------------------------------------------------------------------------------------

assert u(date(0                                )) == u'1970-01-01 01:00:00'
assert u(date(2000, 12, 31                     )) == u'2000-12-31 00:00:00'
assert u(date(2000, 12, 31) - 60 * 60 * 24      ) == u'2000-12-30 00:00:00'
assert   date(2000, 12, 31) .format('%Y-%m-%d'  ) == u'2000-12-31'
assert u(date('Dec 31 2000', format='%b %d %Y' )) == u'2000-12-31 00:00:00'

assert when('Mon Dec 31 2000'                   ) == [u'Mon Dec 31 2000']
assert when('Monday December 31, 2000.'         ) == [u'Monday December 31', u'2000']
assert when('Monday, December 31st 2000'        ) == [u'Monday', u'December 31st 2000']
assert when('Monday 31st of December, 2000'     ) == [u'Monday 31st of December', u'2000']
assert when('2000/12/31'                        ) == [u'2000/12/31']
assert when('23:59 p.m.'                        ) == [u'23:59 p.m.']
assert when('23:59'                             ) == [u'23:59']
assert when('12pm'                              ) == [u'12pm']
assert when('2 weeks ago'                       ) == [u'2 weeks ago']
assert when('5th century'                       ) == [u'5th century']
assert when('100 years'                         ) == [u'100 years']
assert when('1950-2000'                         ) == [u'1950-2000']
assert when('2101 AD'                           ) == [u'2101 AD']
assert when('day one'                           ) == [u'day one']
assert when('the day after tomorrow'            ) == [u'the day after tomorrow']
assert when('the first day of March'            ) == [u'the first day of March']


#---- NET -----------------------------------------------------------------------------------------

g = Graph(
    directed  =True,
    adjacency = {
        1: {2: 1.0}, 
        2: {3: 1.0}, 
        3: {4: 1.0},
        1: {4: 1.0}
    }
)

assert            nodes(g                   )     == {1, 2, 3, 4}
assert           leaves(g                   )     == {1, 2}
assert               sp(g             , 2, 4)     == (2, 3, 4)
assert top( betweenness(g                   ))[0] == 3
assert top(    pagerank(g                   ))[0] == 4

#--------------------------------------------------------------------------------------------------

a = [
    [0, 0, 0, 0, 14],
    [0, 2, 6, 4,  2],
    [0, 0, 3, 5,  6],
    [0, 3, 9, 2,  0],
    [2, 2, 8, 1,  1],
    [7, 7, 0, 0,  0],
    [3, 2, 6, 3,  0],
    [2, 5, 3, 2,  2],
    [6, 5, 2, 1,  0],
    [0, 2, 2, 3,  7]
]

assert round(agreement(a), 2) == 0.38

assert round(mcc({1: {1: 6, 0: 1}, 0: {1: 2, 0: 3}}), 2) == 0.48

#--------------------------------------------------------------------------------------------------

try:
    loc = {r[0]: r for r in csv(next(ls('en-loc.csv')))}

    assert loc['BE'][1] == 'Belgium'
    assert loc['BE'][2] == 'Belgian'
except:
    pass
