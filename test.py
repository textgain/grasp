# encoding: utf-8

from grasp import *

v1 = {'x': 1, 'y': 2}
v2 = {'x': 3, 'y': 4}
v3 = {'x': 5, 'y': 6}
v4 = {'x': 7, 'y': 0}

assert round( distance(v1, v2)              , 2) ==  2.83
assert round(      dot(v1, v2)              , 2) == 11.00
assert round(     norm(v1)                  , 2) ==  2.24
assert round(      cos(v1, v2)              , 2) ==  0.02
assert round(      knn(v1, (v2, v3))[0][0]  , 2) ==  0.98
assert             knn(v1, (v2, v3))[0][1]       ==  v2
assert          sparse(v4)                       == {'x': 7}
assert              tf(v4)                       == {'x': 1, 'y': 0}
assert       features((v1, v2))                  == set(('x', 'y'))
assert next(    tfidf((v1, v2)))['x']            ==  0.25
assert       centroid((v1, v2))                  == {'x': 2, 'y': 3}

assert tokenize(u'(hello) (:'                  ) == u'( hello ) (:'
assert tokenize(u'References [123] (...) :-)'  ) == u'References [123] (...) :-)'
assert tokenize(u'Cats &amp; dogs &#38; etc.'  ) == u'Cats &amp; dogs &#38; etc.'
assert tokenize(u"I'll eat pizza w/ tuna :-)"  ) == u"I 'll eat pizza w/ tuna :-)"
assert tokenize(u'Google (http://google.com)'  ) == u'Google ( http://google.com )'
assert tokenize(u'Also see: www.google.com.'   ) == u'Also see : www.google.com .'
assert tokenize(u'e.g., google.de, google.be'  ) == u'e.g. , google.de , google.be'
assert tokenize(u'One! 😍 Two :) :) Three'     )  == u'One ! 😍\nTwo :) :)\nThree'
assert tokenize(u'Aha!, I see:) Super'         ) == u'Aha ! , I see :)\nSuper'
assert tokenize(u"U.K.'s J. R. R. Tolkien"     ) == u"U.K. 's J. R. R. Tolkien"
assert tokenize(u'your.name@gmail.com!'        ) == u'your.name@gmail.com !'
assert tokenize(u'http://google.com?p=1'       ) == u'http://google.com?p=1'
assert tokenize(u'"Citation." Next p.5'        ) == u'" Citation . "\nNext p. 5'
assert tokenize(u'Hello.Hello! 20:10'          ) == u'Hello .\nHello !\n20:10'
assert tokenize(u'pre-\ndetermined'            ) == u'predetermined'

s1 = u"It/PRON 's/VERB my/PRON fresh/ADJ new/ADJ resumé/NOUN !/PUNC ;)/:)"
s2 = u'1\/2/NUM'

assert list(map(u, chunk(r'^? ?'         , s1))) == [u"It/PRON 's/VERB"]
assert list(map(u, chunk(r'^-'           , s1))) == [u'It/PRON']
assert list(map(u, chunk(r'^PRON'        , s1))) == [u'It/PRON']
assert list(map(u, chunk(r'PRON'         , s1))) == [u'It/PRON', u'my/PRON']
assert list(map(u, chunk(r'BE'           , s1))) == [u"'s/VERB"]
assert list(map(u, chunk(r'VERB'         , s1))) == [u"'s/VERB"]
assert list(map(u, chunk(r'AD[JV]'       , s1))) == [u'fresh/ADJ', u'new/ADJ']
assert list(map(u, chunk(r'ADJ'          , s1))) == [u'fresh/ADJ', u'new/ADJ']
assert list(map(u, chunk(r'ADJ+'         , s1))) == [u'fresh/ADJ new/ADJ']
assert list(map(u, chunk(r'ADJ+ NOUN'    , s1))) == [u'fresh/ADJ new/ADJ resumé/NOUN']
assert list(map(u, chunk(r'ADJ|NOUN+'    , s1))) == [u'fresh/ADJ new/ADJ resumé/NOUN']
assert list(map(u, chunk(u'ADJ re-'      , s1))) == [u'new/ADJ resumé/NOUN']
assert list(map(u, chunk(u're-'          , s1))) == [u'resumé/NOUN']
assert list(map(u, chunk(u'Re-'          , s1))) == [u'resumé/NOUN']
assert list(map(u, chunk(u'-é'           , s1))) == [u'resumé/NOUN']
assert list(map(u, chunk(u'-é/NOUN'      , s1))) == [u'resumé/NOUN']
assert list(map(u, chunk(u'-é/noun'      , s1))) == []
assert list(map(u, chunk(u'-é/VERB'      , s1))) == []
assert list(map(u, chunk(r'NOUN ? ? ?$'  , s1))) == [u'resumé/NOUN !/PUNC ;)/:)']
assert list(map(u, chunk(r'PUNC ?$'      , s1))) == [u'!/PUNC ;)/:)']
assert list(map(u, chunk(r':\) ?$'       , s1))) == [u';)/:)']
assert list(map(u, chunk(r':\)'          , s1))) == [u';)/:)']
assert list(map(u, chunk(r'1/2'          , s2))) == []
assert list(map(u, chunk(r'1\/2'         , s2))) == [u'1\\/2/NUM']
assert list(map(u, chunk(r'1\/2/NUM'     , s2))) == [u'1\\/2/NUM']

assert u(list(constituents(s1))[0][0])           == u'It/PRON'
assert u(list(constituents(s1))[1][0])           == u"'s/VERB"
assert u(list(constituents(s1))[2][0])           == u'my/PRON fresh/ADJ new/ADJ resumé/NOUN'
assert u(list(constituents(s1))[3][0])           == u'!/PUNC'
assert   list(constituents(s1))[0][1]            == u'NP'
assert   list(constituents(s1))[1][1]            == u'VP'
assert   list(constituents(s1))[2][1]            == u'NP'
assert   list(constituents(s1))[3][1]            == u''

e1 = DOM('<div id="main"><div class="story"><p>1</p><p>2</p></div</div>')
e2 = DOM('<div><a href="http://www.site.com">x</a></div>')

assert list(map(u, e1('#main'                ))) == [u(e1)]
assert list(map(u, e1('*#main'               ))) == [u(e1)]
assert list(map(u, e1('div#main'             ))) == [u(e1)]
assert list(map(u, e1('.story'               ))) == [u'<div class="story"><p>1</p><p>2</p></div>']
assert list(map(u, e1('div div'              ))) == [u'<div class="story"><p>1</p><p>2</p></div>']
assert list(map(u, e1('div < p'              ))) == [u'<div class="story"><p>1</p><p>2</p></div>']
assert list(map(u, e1('div > p'              ))) == [u'<p>1</p>', u'<p>2</p>']
assert list(map(u, e1('div div p'            ))) == [u'<p>1</p>', u'<p>2</p>']
assert list(map(u, e1('div p:first-child'    ))) == [u'<p>1</p>']
assert list(map(u, e1('div p:nth-child(1)'   ))) == [u'<p>1</p>']
assert list(map(u, e1('div p:nth-child(2)'   ))) == [u'<p>2</p>']
assert list(map(u, e1('p:not(":first-child")'))) == [u'<p>2</p>']
assert list(map(u, e1('div p:contains("2")'  ))) == [u'<p>2</p>']
assert list(map(u, e1('p + p'                ))) == [u'<p>2</p>']
assert list(map(u, e1('p ~ p'                ))) == [u'<p>2</p>']
assert list(map(u, e2('*[href]'              ))) == [u'<a href="http://www.site.com">x</a>']
assert list(map(u, e2('a[href^="http://"]'   ))) == [u'<a href="http://www.site.com">x</a>']
assert list(map(u, e2('a[href$=".com"]'      ))) == [u'<a href="http://www.site.com">x</a>']
assert list(map(u, e2('a[href*="site"]'      ))) == [u'<a href="http://www.site.com">x</a>']

assert plaintext(e2                            ) == u'x'
assert plaintext(e2, keep={'a': []}            ) == u'<a>x</a>'
assert plaintext(e2, keep={'a': ['href']}      ) == u'<a href="http://www.site.com">x</a>'
assert plaintext(e2, ignore='a'                ) == u''
