# Grasp.py – knowledge base

**Grasp** has a knowledge base of optional CSVs with useful world knowledge. These can be used in NLP tasks (e.g., guess region) or ML models (e.g., guess gender).

## en-loc.csv

Essential geographical data about world **countries** and **states**:

|  #  | FIELD   | DESCRIPTION                                         |
| :-: | :------ | :-------------------------------------------------- |
| 0   | code    | country code: _GB_, _DE_, _FR_, ... ([ISO 3166-2](https://en.wikipedia.org/wiki/ISO_3166-2)) |
| 1   | name    | country name: _United Kingdom_, _Germany_, _France_ |
| 2   | who     | country people: _British_, _German_, _French_       |
| 3   | where   | country region: _Europe_                            |
| 4   | what    | country government: _country (monarchy)_            |
| 5   | city    | country capital: _London_, _Berlin_, _Paris_        |
| 6   | lang    | country language(s): _en_, _de_, _fr_               |
| 6   | flag    | country flag: 🇬🇧, 🇩🇪, 🇫🇷                           |

## en-nom.csv

Essential demographical data about **names** of people: <sup>[1](https://www.heise.de/ct/ftp/07/17/182/)</sup>

|  #  | FIELD   | DESCRIPTION                                         |
| :-: | :------ | :-------------------------------------------------- |
| 0   | name    | given name: _Alice_, _Bob_, ...                     |
| 1   | gender  | most common gender: _f_, _m_                        |
| 2   | where   | most common countries: _NL_, _US_, _IE_             |
| 3   | when    | most births: _1920_, _1940_                         |

To train a model that predicts gender by name, we can map each known name to a dict of features like suffix and letter pairs. For _Alice_ &rarr; _al_, _li_, _ic_, _ce_, _ice_, _lice_, ... The model can learn from these features to predict the gender of unknown names with ~80% accuracy. For example, _Elice_ (_el_, _li_, _ic_, ...) resembles _Alice_ and is probably `f`.

```py
from grasp import csv, fit, vec, top
```
```py
def v(s):
    s = s.lstrip('@#')
    s = s.lower()
    v = vec(s, features=('c2', 'c3', 'c4', '$1', '$2', '$3'))
    return v
```
```py
data = []
for name, gender, where, when in csv('en-nom.csv'):
    data.append((v(name), gender))

m = fit(data) # P 78% R 79%
```
```py
print(top(m.predict(v('Aragorn')))) # m
```