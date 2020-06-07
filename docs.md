#Grasp.py – Explainable AI

**Grasp** is a lightweight Explainable AI toolkit for Python 2 & 3, with building blocks for data mining, natural language processing (NLP), machine learning (ML) and network analysis. It has 300+ free, fast, clean, concise implementations of classic algorithms, with only ~25 lines of code per function, self-explanatory names, no external depencies, bundled into one well-documented file: [grasp.py](https://github.com/textgain/grasp) (200KB).

**Grasp** is developed and used by [Textgain](https://textgain.com), a language tech company that uses open-source intelligence (OSINT) for societal good.

##Do data mining

**Download stuff** with `download(url)` (or `dl`), with built-in caching and logging:

```python
src = dl('https://www.textgain.com', cached=True)
```

**Parse HTML** with `DOM(html)` into an `Element` tree and search it with [CSS Selectors](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Selectors):

```py
for e in DOM(src)('a[href^="http"]'): # external links
    print(e.href)
```

**Strip HTML** with `plaintext(DOM)` (or `plain`) to just get a text string:

```py
for word, count in wc(plain(DOM(src))).items():
	print(word, count)
```

**Find tweets** with `twitter.seach(str)`:

```py
for tweet in first(10, twitter.search('from:textgain')): # latest 10
    print(tweet.date, tweet.text)
```

**Deploy APIs** with `App`. It's friends with WSGI and Nginx:

```py
app = App() # check http://127.0.0.1:8080/app?q=cat
@app.route('/')
def index(*path, **query):
    return 'Hello from %s %s' % (path, query)
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
```

##Do NLP

**Find words & sentences** with `tokenize(str)` (or `tok`), at ~125K words/sec:

```py
print(tok("Mr. etc. aren't sentence breaks! ;) This is.", 'en'))
```

**Find word types** with `tag(str)`, in 10+ languages using robust ML models from [UD](https://universaldependencies.org):

```py
for word, pos in tag(tok("Mr. etc. aren't sentence breaks! ;)"), 'en'):
    print(word, pos)
```

Part-of-speech tags (POS) include `NOUN`, `VERB`, `ADJ` (adjective), `ADV` (adverb), `DET` (determiner) `PRON` (pronoun), `PREP` (preposistion), with up to 97% predictive accuracy.

**Find word intent** with `sentiment(str)`. Is the author being positive or negative?

```py
print(sentiment('Awesome stuff! 😁')) # +0.5
print(sentiment('Horrible crap! 😡')) # -1.0
```

##Do ML

Machine Learning (ML) means algorithms that *learn by example* from humans, like showing them 10,000 spam messages and 10,000 real emails. The ML will then predict if new messages are spam or not. For example, it might'e learned that *lottery + winner* is statistically more likely to occur in spam – it's basically doing math on word counts fed by its teacher.

**Count words** with `wc(str)` into a `{word: count}` dict:

**Count word pairs** with `ngrams(str)`:

**Count word parts** with `ch(str)`:

**Quantify text** with `vectorize(str)` (or `vec`) into a `{feature: weight}` dict:

```py
v1 = vec('I love cats! 😀', features=['c3', 'w1'])
v2 = vec('I hate cats! 😡', features=['c3', 'w1'])
```

Features include `c1`, `c2`, `c3` (all 3 consecutive characters: *loved* → *lov*, *ove*, *ved*), `w1`, `w2`, `w3` (all 3 consecutive words) and `%` (style; average word length etc.).

Grasp is a lightweight Python toolkit for data mining (search engines, servers, HTML DOM + CSS selectors, plaintext), Natural Language Processing (NLP; tokenization, multilingual part-of-speech tagging & sentiment analysis), Machine Learning (ML; clustering, classification, confusion matrix, n-grams),

