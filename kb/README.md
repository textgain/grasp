# Grasp.py – knowledge base

## en-loc.csv

| Index | Field   | Description                                         |
| :---: | :------ | :-------------------------------------------------- |
| 0     | code    | country code: _GB_, _DE_, _FR_, ... ([ISO 3166-2](https://en.wikipedia.org/wiki/ISO_3166-2)) |
| 1     | name    | country name: _United Kingdom_, _Germany_, _France_ |
| 2     | adj     | country demonym: _British_, _German_, _French_      |
| 3     | region  | country region: _Europe_                            |
| 4     | gov     | country government: _country (monarchy)_            |
| 5     | city    | country capital: _London_, _Berlin_, _Paris_        |
| 6     | lang    | country language(s): _en_, _de_, _fr_               |

## en-nom.csv

| Index | Field   | Description                                         |
| :---: | :------ | :-------------------------------------------------- |
| 0     | name    | first name: _Alice_, _Bob_                          |
| 1     | gender  | most frequent gender: _f_, _m_                      |
| 2     | where   | most frequent countries: _NL_, _US_, _IE_           |
| 3     | when    | most frequent births: _1920_, _1940_                |

```py
from grasp import vec, fit, top, csv, cd
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
for name, gender, where, when in csv(cd('kb', 'en-nom.csv')):
    data.append((v(name), gender))

m = fit(data) # P 78% R 79%
```
```py
print(top(m.predict(v('Aragorn')))) # m 75%
```
