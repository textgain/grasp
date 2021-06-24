# encoding: utf-8

##### SVG.PY ######################################################################################

__version__   =  '1.1'
__license__   =  ''
__credits__   = ['Tom De Smedt', 'Guy De Pauw']
__email__     =  'info@textgain.com'
__author__    =  'Textgain'
__copyright__ =  'Textgain'

###################################################################################################

import sys
import inspect
import colorsys
import codecs
import unicodedata
import hashlib
import mimetypes
import base64
import struct
import random; _random=random
import math

PY2 = sys.version.startswith('2')

#--------------------------------------------------------------------------------------------------

def data_uri(path, default='application/octet-stream'):
    """ Returns the data URI string for the given file.
    """
    type = mimetypes.guess_type(path)[0] or default
    s = open(path, 'rb').read()
    s = base64.b64encode(s)
    s = s.decode('utf-8')
    s = 'data:%s;base64,%s' % (type, s)
    return s

#---- GEOMETRY ------------------------------------------------------------------------------------

from math import sqrt, atan2, cos, sin, degrees, radians

def angle(x0, y0, x1, y1):
    """ Returns the angle between two points.
    """
    return degrees(atan2(y1 - y0, x1 - x0))

def distance(x0, y0, x1, y1):
    """ Returns the distance between two points.
    """
    return sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

def coordinates(x, y, distance, angle):
    """ Returns the point at distance and angle from x, y.
    """
    return x + cos(radians(angle)) * distance, \
           y + sin(radians(angle)) * distance

def polar(x, y):
    """ Returns the point as a (distance, angle) vector.
    """
    return distance(0, 0, x, y), \
              angle(0, 0, x, y)

def lerp(t, x0, y0, x1, y1):
    """ Returns the point at t (0.0-1.0) on the line between two points.
    """
    # linear interpolation
    return (1 - t) * x0 + x1 * t, \
           (1 - t) * y0 + y1 * t

def berp(t, x0, y0, x1, y1, x2, y2, x3, y3, split=False):
    """ Returns the point at t (0.0-1.0) on the Bézier curve.
    """
    # repeat interpolation (De Casteljau)
    ax, ay = lerp(t, x0, y0, x1, y1)
    bx, by = lerp(t, x1, y1, x2, y2)
    cx, cy = lerp(t, x2, y2, x3, y3)
    dx, dy = lerp(t, ax, ay, bx, by)
    ex, ey = lerp(t, bx, by, cx, cy)
    fx, fy = lerp(t, dx, dy, ex, ey)

    if split:
        return (dx, dy, ex, ey, fx, fy), (ax, ay, cx, cy)
    else:
        return (dx, dy, ex, ey, fx, fy)

def smoothstep(v, min=0.0, max=1.0):
    """ Returns a smooth transition (0.0-1.0) for v between min and max.
    """
    # polynomial interpolation (Hermite)
    if v < min: 
        return min
    if v > max: 
        return max
    else:
        v = float(v - min) / (max - min)
        v = v * v * (3 - 2 * v)
        return v

def clamp(v, min=0.0, max=1.0):
    """ Returns the value between min and max.
    """
    if v < min:
        return min
    if v > max:
        return max
    else:
        return v

def zoom(points=[], radius=1.0):
    """ Returns the scaled list of points.
    """
    p = [polar(x, y) for x, y in points]
    r = radius / max(d for d, a in p)
    p = [(d * r, a) for d, a in p]
    p = [coordinates(0, 0, d, a) for d, a in p]
    return p

#---- ITERATION -----------------------------------------------------------------------------------

def polyline(x0, y0, x1, y1, x2, y2, x3, y3, n=10):
    """ Returns an iterator of n+1 points (i.e., n line segments).
    """
    for t in range(n + 1):
        yield berp(float(t) / n, x0, y0, x1, y1, x2, y2, x3, y3)[-2:]

def pairwise(a):
    """ Returns an iterator of consecutive pairs.
    """
    # pairwise((1, 2, 3)) => (1, 2), (2, 3)
    for v in a:
        try: 
            yield prev, v
        except UnboundLocalError:
            pass
        prev = v

def cumulative(a):
    """ Returns an iterator of the cumulative sum.
    """
    # cumulative((1, 2, 3)) => 1, 3, 6
    n = 0
    for v in a:
        n += v
        yield n

def relative(a):
    """ Returns an iterator of relative values (sum 1).
    """
    # relative((1, 2, 3)) => 0.16, 0.33, 0.5
    a = list(a)
    n = sum(a)
    n = float(n) or 1
    for v in a:
        yield v / n

#---- NUMBERS -------------------------------------------------------------------------------------
# The number formatting functions are useful for chart visualizations.

class short(str):
    """ Returns the given number as a short string.
    """
    # short(1234) => 1.2K
    def __new__(cls, v, precision=1, format=None):
        if v >= 1000 or type(v) is float:
            f = '%.*f'.replace('*', str(precision))
        else:
            f = '%i'
        if v <  1000:
            n = 1, ''
        if v <= 1 and v >= 0 and format == '%': # 0.1 => 10.0%
            n = 1e-2, '%'
        if v >= 1e+3 - 1e+2:
            n = 1e+3, 'K'
        if v >= 1e+6 - 1e+5:
            n = 1e+6, 'M'
        if v >= 1e+9 - 1e+8:
            n = 1e+9, 'B'

        s = v / n[0]
        s = f % s
        s = s + n[1]
        s = str.__new__(cls, s)  # '1.1K'
        s.v = v                  #  1100
        s.n = n[0]               #  1000.0
        s.u = n[1]               # 'K'
        s.p = precision          #  1
        s.f = format             #  None
        return s

    def __add__(self, v):
        return short(self.v + v, self.p, self.f)

    def __sub__(self, v):
        return short(self.v - v, self.p, self.f)

    def __int__(self):
        return int(float(self))

    def __float__(self):
        return self.v * self.n

def ceil(v, m=1.0):
    """ Returns the upper rounded float,
        by order of magnitude.
    """
    # ceil(123, 1.0) => 200
    # ceil(123, 0.5) => 150
    v = v / m
    n = abs(v) or 1
    n = math.log10(n)
    v = math.pow(.1, int(n)) * v
    v = math.ceil(v)
    v = math.pow(10, int(n)) * v
    v = v * m
    return v

def floor(v, m=1.0):
    """ Returns the lower rounded float,
        by order of magnitude.
    """
    # floor(123, 1.0) => 100
    v = v / m
    n = abs(v) or 1
    n = math.log10(n)
    v = math.pow(.1, int(n)) * v
    v = math.floor(v)
    v = math.pow(10, int(n)) * v
    v = v * m
    return v

def format(v):
    """ Returns a number string format,
        by order of magnitude.
    """
    if type(v) is short:
        return '%s'
    if float(v).is_integer():
        return '%.0f'
    if abs(v) > 100:
        return '%.0f' # 0, 100, 200, ...
    if abs(v) > 2.0:
        return '%.1f' # 0, 0.1, 0.2, ...
    else:
        return '%.2f'

def steps(v1, v2):
    """ Returns steps from int v1 to v2.
    """
    r = abs(v2 - v1)
    r = str(r).strip('0.')
    r = int(r)

    if r == 1:
        return 5
    if not r % 5: # 0, 5, 10, 15, 20, 25
        return 5
    if not r % 4: # 0, 6, 12, 18, 24
        return 4
    if not r % 3: # 0, 7, 14, 21
        return 3
    else:
        return 2

#---- STATISTICS ----------------------------------------------------------------------------------

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
    return sqrt(sum((v - m) ** 2 for v in a) / n)

def peaks(a, z=1):
    """ Returns a list of indices of values that are
        more than z standard deviations above the mean.
    """
    a = list(a)
    m = avg(a)
    s = sd(a)
    a = ((v - m) / s for v in a)
    a = [i for i, v in enumerate(a) if v > z]
    return a

# print(peaks([0, 0, 0, 10, 100, 1, 0], z=1))

#---- COLOR ---------------------------------------------------------------------------------------
# The Color object represents shape fill and stroke (outline) colors in terms of R, G, B, A values.
# The Color object can also be created with H, S, B values (i.e., hue, saturation, brightness), and
# rotated on the RYB color wheel (painter's model) to find aesthetical color combinations.

RGB = 'rgb'
HSB = 'hsb'

RYB = (
    0.000, #  0° = H 0.00
    0.022, # 15° = H 0.02
    0.047, # 30° = H 0.05
    0.072, # 45° = H 0.07 ...
    0.094,
    0.114,
    0.133,
    0.150,
    0.167,
    0.225,
    0.286,
    0.342,
    0.383,
    0.431,
    0.475,
    0.519,
    0.567,
    0.608,
    0.65 ,
    0.697,
    0.742,
    0.783,
    0.828,
    0.914,
    1.000,
)

class Color(object):

    def __init__(self, *v, **k):
        """ A color with RGBA values between 0.0-1.0.
        """
        if v and isinstance(v[0], (list, tuple)):        # Color(list)
            v = v[0]
        if len(v) == 0:                                  # Color()
            r, g, b, a = 0, 0, 0, 0
        elif len(v) == 1 and v[0] is None:               # Color(None)
            r, g, b, a = 0, 0, 0, 0
        elif len(v) == 1 and isinstance(v[0], Color):    # Color(Color)
            r, g, b, a = v[0].r, v[0].g, v[0].b, v[0].a
        elif len(v) == 1:                                # Color(k)
            r, g, b, a = v[0], v[0], v[0], 1
        elif len(v) == 2:                                # Color(k, a)
            r, g, b, a = v[0], v[0], v[0], v[1]
        elif len(v) == 3:                                # Color(r, g, b)
            r, g, b, a = v[0], v[1], v[2], 1
        elif len(v) == 4:                                # Color(r, g, b, a)
            r, g, b, a = v[0], v[1], v[2], v[3]
        if k.get('mode') == HSB:                         # Color(h, s, b, a, mode=HSB)
            r, g, b = colorsys.hsv_to_rgb(r, g, b)

        self.r = float(r)
        self.g = float(g)
        self.b = float(b)
        self.a = float(a)

    def __eq__(self, clr):
        return self.rgba == clr.rgba

    @property
    def rgb(self):
        return self.r, self.g, self.b

    @property
    def rgba(self):
        return self.r, self.g, self.b, self.a

    @property
    def hsba(self):
        return colorsys.rgb_to_hsv(self.r, self.g, self.b) + (self.a,)

    def rotate(self, angle=180):
        """ Returns the color rotated on the RYB color wheel.
        """
        # 1) Get the angle of the hue on the RYB color wheel:
        h, s, b, a = self.hsba
        i = next(i for i, v in enumerate(RYB) if v > h)
        x = RYB[i-1]
        y = RYB[i]
        t = (h - x) / (y - x)
        d = angle + i * 15 - 15 * (1 - t)
        # 2) Get the hue at the new angle:
        i = int(d // 15)
        x = RYB[i]
        y = RYB[i+1]
        t = d % 15 / 15
        h = x + t * (y - x)
        return Color(h, s, b, a, mode=HSB)

    def __str__(self):
        return 'rgba(%i, %i, %i, %.2f)' % (
            self.r * 255, 
            self.g * 255, 
            self.b * 255,
            self.a
        )

    def __repr__(self):
        return 'Color(r=%.2f, g=%.2f, b=%.2f, a=%.2f)' % (
            self.r, 
            self.g, 
            self.b, 
            self.a
        )

    def __iter__(self):
        return iter((
            self.r, 
            self.g, 
            self.b, 
            self.a
        ))

TRANSPARENT, BLACK, WHITE = Color(0, 0), Color(0), Color(1)

def luminance(clr):
    """ Returns the perceived brightness of the color.
    """
    return clr.r * 0.3 + clr.g * 0.6 + clr.b * 0.1 # Y

def complement(clr):
    """ Returns the complementary color.
    """
    return clr.rotate(180)

def adjust(clr, h=0.0, s=0.0, b=0.0, a=0.0):
    """ Returns the adjusted color.
    """
    return Color(*(clamp(v1 + v2) for v1, v2 in zip(clr.hsba, (h, s, b, a))), mode=HSB)

def mix(t, clr1, clr2, **k):
    """ Returns an interpolated color between the given colors at t (0.0-1.0).
    """
    return Color(*((1 - t) * v1 + v2 * t for v1, v2 in zip(clr1, clr2)), **k)

class Gradient(list):

    def __init__(self, clr1, clr2, *clr3):
        """ A smooth transition between the given colors.
        """
        list.__init__(self, (clr1, clr2) + clr3)

    def __call__(self, n=100):
        """ Returns an iterator of n interpolated colors.
        """
        for i in range(n):
            try:
                yield self.at(t=i / float(n - 1))
            except ZeroDivisionError:
                yield self.at(t=0.5)

    def at(self, t):
        """ Returns an interpolated color at t (0.0-1.0).
        """
        if t <= 0:
            return self[0]
        if t >= 1:
            return self[-1]

        n = len(self) - 1
        t = divmod(t, 1.0 / n)
        i = int(t[0])
        t = n * t[1]
        return mix(t, self[i], self[i+1])

def mono(clr):
    """ Returns a monochromatic gradient.
    """
    h, s, b, a = clr.hsba
    s  = s * 0.6
    s  = s + 0.3
    g1 = Color(h, 1.00, 0.30, a, mode=HSB) # deep
    g2 = Color(h, s   , 0.90, a, mode=HSB)
    g3 = Color(h, 0.30, 1.00, a, mode=HSB) # pale
    g  = Gradient(g1, g2, g3)
    return g

def swatch(clr, n=10, f=mono):
    """ Returns a list of colors.
    """
    if isinstance(clr, Color):
        clr = f(clr)
    if isinstance(clr, Gradient):
        clr = list(clr(n))
    if isinstance(clr, list):
        clr = list(clr) + clr[-1:] * (n - len(clr)) # repeat last
    return clr

#---- FONT ----------------------------------------------------------------------------------------
# The font table defines the width of ASCII characters at size 100px, for a select number of fonts:

EMOJI = {
    u'😊', u'☺️', u'😉', u'😌', u'😏', u'😎', u'😍', u'😘', u'😴', u'😀', u'😃', u'😄', 
    u'😅', u'😇', u'😂', u'😭', u'😢', u'😱', u'😳', u'😬', u'😜', u'😛', u'😁', u'😐', 
    u'😕', u'🤔', u'😧', u'😦', u'😒', u'😞', u'😔', u'😫', u'😩', u'😠', u'😡', u'🤢', 
    u'❤️', u'💔', u'💘', u'💕', u'👍', u'👎', u'🙏', u'👊', u'🖕', u'☝️', u'🔫', u'💣', 
}

CHARS = {ch: i for i, ch in enumerate(map(chr, range(32, 127)))} # {' ': 0, '!': 1, ...}

FONTS = {
    #       !   "   #   $   %   &   '   (   )   *   +   ,   -   .   /   0   1   2   3   4   
    ('Arial', 'normal'): [
        28, 28, 36, 56, 56, 89, 67, 19, 33, 33, 39, 58, 28, 33, 28, 28, 56, 56, 56, 56, 56, 
        56, 56, 56, 56, 56, 28, 28, 58, 58, 58, 56, 99, 67, 67, 72, 72, 67, 61, 78, 72, 28, 
        50, 67, 56, 83, 72, 78, 67, 78, 72, 67, 61, 72, 67, 95, 67, 67, 61, 28, 28, 28, 47, 
        56, 33, 56, 56, 50, 56, 56, 28, 56, 56, 22, 22, 50, 22, 83, 56, 56, 56, 56, 33, 50, 
        28, 56, 50, 72, 50, 50, 50, 33, 26, 33, 58, 0], 
    ('Arial', 'bold'): [
        28, 33, 48, 56, 56, 89, 72, 24, 33, 33, 39, 58, 28, 33, 28, 28, 56, 56, 56, 56, 56, 
        56, 56, 56, 56, 56, 33, 33, 58, 58, 58, 61, 98, 72, 72, 72, 72, 67, 61, 78, 72, 28, 
        56, 72, 61, 83, 72, 78, 67, 78, 72, 67, 61, 72, 67, 95, 67, 67, 61, 33, 28, 33, 58, 
        56, 33, 56, 61, 56, 61, 56, 33, 61, 61, 28, 28, 56, 28, 89, 61, 61, 61, 61, 39, 56, 
        33, 61, 56, 78, 56, 56, 50, 39, 28, 39, 58, 0], 
    ('Times New Roman', 'normal'): [
        25, 33, 41, 50, 50, 83, 78, 18, 33, 33, 50, 56, 25, 33, 25, 28, 50, 50, 50, 50, 50, 
        50, 50, 50, 50, 50, 28, 28, 56, 56, 56, 45, 92, 72, 67, 67, 72, 61, 56, 72, 72, 33, 
        39, 72, 61, 89, 72, 72, 56, 72, 67, 56, 61, 72, 72, 95, 72, 72, 61, 33, 28, 33, 47, 
        50, 33, 45, 50, 45, 50, 45, 33, 50, 50, 28, 28, 50, 28, 78, 50, 50, 50, 50, 33, 39, 
        28, 50, 50, 72, 50, 50, 45, 48, 20, 48, 54, 0], 
    ('Times New Roman', 'bold'): [
        25, 33, 56, 50, 50, 99, 83, 28, 33, 33, 50, 57, 25, 33, 25, 28, 50, 50, 50, 50, 50, 
        50, 50, 50, 50, 50, 33, 33, 57, 57, 57, 50, 93, 72, 67, 72, 72, 67, 61, 78, 78, 39, 
        50, 78, 67, 95, 72, 78, 61, 78, 72, 56, 67, 72, 72, 99, 72, 72, 67, 33, 28, 33, 58, 
        50, 33, 50, 56, 45, 56, 45, 33, 50, 56, 28, 33, 56, 28, 83, 56, 50, 56, 56, 45, 39, 
        33, 56, 50, 72, 50, 50, 45, 40, 22, 40, 52, 0], 
}

def textsize(s, fontname, fontsize, fontweight='normal'):
    """ Returns the approximate (width, height) of the given string.
    """
    fontsize = int(fontsize) # 10.5 => '11px'

    if isinstance(s, u''.__class__):
        f = unicodedata.combining
        b = unicodedata.normalize('NFKD', s) # 'touché' => 'touche´'
        b = ''.join(c for c in b if not f(c))
        b = b.replace(u'“', '"')
        b = b.replace(u'”', '"')
        b = b.replace(u'‘', "'")
        b = b.replace(u'’', "'")
        b = b.replace(u'ø', 'o') 
        b = b.encode('ascii', 'replace')
        b = b.decode('utf-8')
    else:
        b = s # # PY2 str

    b = b.replace('\n', ' ')

    m = FONTS[fontname, fontweight]
    w = fontsize * 0.01 * sum(m[CHARS.get(ch, -1)] for ch in b) + \
        fontsize * 1.00 * sum(1 for e in EMOJI if e in s)
    h = fontsize
    return w, h

# print(textsize(u'touché 😁', 'Arial', 10))

def truncate(s, width, font, fontsize, fontweight='normal', placeholder='...'):
    """ Returns the string up to the given width.
    """
    f = lambda s: textsize(s, font, fontsize, fontweight)[0] > width

    while s != placeholder and f(s):
        s = s[:-len(placeholder)]
        s = s[:-1]
        s = s + placeholder
    if not f(s):
        return s
    else:
        return ''

# print truncate('supercalifragilisticexpialidocious', 100, 'Arial', 12)

#---- BÉZIER PATH ---------------------------------------------------------------------------------

MOVETO, LINETO, CURVETO, CLOSE = 'moveto', 'lineto', 'curveto', 'close'

class Point(object):
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __iter__(self):
        return iter((
            self.x,
            self.y
        ))

class PathElement(Point):

    def __init__(self, cmd, *pt):
        """ A point in a path, with control points for curve segments.
        """
        if cmd == MOVETO:
            pt = pt * 3
        if cmd == LINETO:
            pt = pt * 3
        if cmd == CLOSE:
            pt = (0, 0, 0, 0, 0, 0)

        self.cmd = cmd
        self.ctrl1 = Point(pt[0], pt[1])
        self.ctrl2 = Point(pt[2], pt[3])
        self.x = pt[4]
        self.y = pt[5]

    def __str__(self):
        if self.cmd == MOVETO:
            return 'M%.1f,%.1f' % (self.x, self.y)
        if self.cmd == LINETO:
            return 'L%.1f,%.1f' % (self.x, self.y)
        if self.cmd == CURVETO:
            return 'C%.1f,%.1f,%.1f,%.1f,%.1f,%.1f' % tuple(self)
        if self.cmd == CLOSE:
            return 'Z'

    def __repr__(self):
        return 'PathElement(cmd=%s, x=%.1f, y=%.1f)' % (
            self.cmd.upper(), 
            self.x, 
            self.y
        )

    def __iter__(self):
        return iter((
            self.ctrl1.x,
            self.ctrl1.y,
            self.ctrl2.x,
            self.ctrl2.y,
            self.x,
            self.y
        ))

class BezierPath(list):

    def moveto(self, x, y):
        """ Adds a new point to the path at x, y.
        """
        self.append(PathElement(MOVETO, x, y))

    def lineto(self, x, y):
        """ Adds a line from the previous point to x, y.
        """
        self.append(PathElement(LINETO, x, y))

    def curveto(self, ctrl1x, ctrl1y, ctrl2x, ctrl2y, x, y):
        """ Adds a curve from the previous point to x, y, 
            with control points that define the curvature.
        """
        self.append(PathElement(CURVETO, ctrl1x, ctrl1y, ctrl2x, ctrl2y, x, y))

    def close(self):
        """ Adds a line from the previous point to the last MOVETO.
        """
        self.append(PathElement(CLOSE))

    def rect(self, x, y, w, h):
        """ Adds a rectangle to the path.
        """
        self.moveto(x, y)
        self.lineto(x + w, y)
        self.lineto(x + w, y + h)
        self.lineto(x, y + h)
        self.lineto(x, y)

    def ellipse(self, x, y, w, h): 
        """ Adds an ellipse to the path.
        """
        w = w * 0.5
        h = h * 0.5
        i = w * 0.552 # 4 / 3 * (√2 - 1)
        j = h * 0.552
        self.moveto(x - w, y)
        self.curveto(x - w, y - j, x - i, y - h, x, y - h)
        self.curveto(x + i, y - h, x + w, y - j, x + w, y)
        self.curveto(x + w, y + j, x + i, y + h, x, y + h)
        self.curveto(x - i, y + h, x - w, y + j, x - w, y)
        self.close()

    @property
    def length(self):
        """ Returns the length of the path.
        """
        return sum(self.lengths)

    @property
    def lengths(self):
        """ Returns the length of the path segments.
        """
        for pt in self:
            if pt.cmd == MOVETO:
                yield 0.0; o=pt
            if pt.cmd == LINETO:
                yield distance(x, y, pt.x, pt.y)
            if pt.cmd == CURVETO: # rectification
                yield sum(distance(*(pt1 + pt2)) for pt1, pt2 in pairwise(polyline(x, y, *pt, n=20)))
            if pt.cmd == CLOSE:
                yield distance(x, y, o.x, o.y)
            if pt.cmd != CLOSE:
                x = pt.x
                y = pt.y

    def point(self, t, _lengths=None):
        """ Returns a synthetic PathElement at t (0.0-1.0) on the path.
        """
        for (t1, pt1), (t2, pt2) in segmented(self, _lengths):
            if pt1.cmd == MOVETO:
                o = pt1
            if pt2.cmd == MOVETO:
                o = pt2
            if t1 <= t <= t2:
                t = (t - t1) / (t2 - t1 or 1)
                if pt2.cmd == LINETO:
                    return PathElement(pt2.cmd, *lerp(t, pt1.x, pt1.y, pt2.x, pt2.y))
                if pt2.cmd == CURVETO:
                    return PathElement(pt2.cmd, *berp(t, pt1.x, pt1.y, *pt2))
                if pt2.cmd == CLOSE:
                    return PathElement(pt2.cmd, *lerp(t, pt1.x, pt1.y, o.x, o.y))

    def points(self, n=10, t1=0.0, t2=1.0):
        """ Returns an iterator of n synthetic PathElements on the path.
        """
        cached = list(self.lengths)

        for t in range(n):
            # n=1: t 0.0
            # n=2: t 0.0, 1.0
            # n=3: t 0.0, 0.5, 1.0
            t = t / float(n - 1 if n > 1 else n)
            t = t * (t2 - t1) + t1
            if 0 <= t <= 1:
                yield self.point(t, _lengths=cached)
            else:
                return

    def __repr__(self):
        return 'BezierPath(length=%i)' % self.length

Path = BezierPath

def segmented(path, _lengths=None):
    """ Returns an iterator of consecutive (t1, pt1), (t2, pt2) pairs,
        where t1 and t2 is the relative start and end of each segment.
    """
    p = _lengths or path.lengths # (100, 300, 600)
    p = relative(p)              # (0.1, 0.3, 0.6)
    p = cumulative(p)            # (0.1, 0.4, 1.0)
    p = zip(p, path)
    p = pairwise(p)
    return p

# p = Path()
# p.moveto(100, 100)
# p.lineto(200, 200)
# p.lineto(300, 300)
# p.close()
# drawpath(p, stroke=color(0), strokewidth=1)

# for pt in p.points(p):
#     ellipse(pt.x, pt.y, 3, 3)

def fit(points=[], k=0.5):
    """ Returns a path from the list of (x, y) tuples, with curvature k.
    """
    p = Path()
    a = list(map(tuple, points))
    k = max(k, 0.0)
    k = min(k, 1.0) * 100

    if len(a) > 0:
        dx0, dy0 = a[0]
        p.moveto(dx0, dy0)
    for (x0, y0), (x1, y1), (x2, y2) in zip(a, a[1:], a[2:]):
        r1 = angle(x1, y1, x0, y0)
        r2 = angle(x1, y1, x2, y2)
        r  = (r2 + r1) / 2
        w  = (r2 < r1) and -1 or +1
        dx1, dy1 = coordinates(x1, y1, k, r - 90 * w)
        p.curveto(dx0, dy0, dx1, dy1, x1, y1)
        dx0, dy0 = coordinates(x1, y1, k, r + 90 * w)
    if len(a) > 1:
        p.curveto(dx0, dy0, *a[-1] * 2)

    return p

#---- CONTEXT -------------------------------------------------------------------------------------

WIDTH   = 1000.0
HEIGHT  = 1000.0 

LEFT    = 0
RIGHT   = 1
CENTER  = 2
JUSTIFY = 3

NORMAL  = 'normal'
BOLD    = 'bold'
ITALIC  = 'italic'

NONZERO = 'nonzero'
EVENODD = 'evenodd'

attributes = {
    'id'          : ('id'               , '%s'  ),
    'type'        : ('class'            , '%s'  ),
    'fill'        : ('fill'             , '%s'  ),
    'stroke'      : ('stroke'           , '%s'  ),
    'strokewidth' : ('stroke-width'     , '%.1f'),
    'strokestyle' : ('stroke-dasharray' , '%s'  ),
    'font'        : ('font-family'      , '%s'  ),
    'fontsize'    : ('font-size'        , '%ipx'),
    'fontweight'  : ('font-weight'      , '%s'  ),
}

def serialize(**attrs):
    """ Returns the XML-formatted attribute string.
    """
    a = []
    for k, v in attrs.items():
        if k == 'fill'   and not isinstance(v, Color):
            v = Color(v)
        if k == 'stroke' and not isinstance(v, Color):
            v = Color(v)
        if k == 'strokestyle':
            v = ' '.join(map(str, v))
        if k in attributes:
            v = attributes[k][1] % v
            k = attributes[k][0]
            a.append('%s="%s"' % (k, v))
    return ' '.join(a)

# print(serialize(font='Arial', fontsize=100, fill=Color(0)))

def mixin(s, **attrs):
    """ Returns the XML element with the attributes mixed in.
    """
    if attrs:
        i = s.find('>')
        i = i - 1 if i > 0 and s[i-1] == '/' else i # '/>'
        i = i - 1 if i > 0 and s[i-1] == ' ' else i
        s = '%s %s%s' % (s[:i], serialize(**attrs), s[i:])
    if attrs.get('link'):
        s = '<a href="%s">%s</a>' % (encode(attrs['link']), s)
    return s

# print(mixin("<text>blah</text>", fontsize=100, link='...'))

def encode(s):
    """ Returns the XML-encoded string.
    """
    s = s.replace('&', '&amp;' )
    s = s.replace('>', '&gt;'  )
    s = s.replace('<', '&lt;'  )
    s = s.replace('"', '&quot;')
    s = s.replace("'", '&apos;')
    return s

# print(encode('<url>'))

XML = '<?xml version="1.0" encoding="utf-8"?>'

class Context(list):

    def __init__(self, width=WIDTH, height=HEIGHT, _global=False):
        self.width   = width
        self.height  = height
        self._global =_global
        self._defs   = {}
        self._stack  = [[]] # push/pop
        self._state  = {
            'background' : Color(1),
                  'path' : Path(),
                  'font' : ['Arial', 12, 'normal'],
        }

    def size(self, w, h):
        """ Resizes the canvas to width w and height h.
        """
        self.width  = w
        self.height = h

        if self._global:
            # Update global variables:
            g = inspect.currentframe()
            g = g.f_back
            g = g.f_globals
            g['WIDTH' ] = w
            g['HEIGHT'] = h

    def push(self):
        """ Pushes the drawing state.
            Subsequent translate(), rotate(), scale() and apply until pop().
        """
        # for i in range(10):
        #     push()
        #     translate(i * 100, 100)
        #     rotate(i * 36)
        #     rect(-50, -50, 100, 100)
        #     pop()))
        self.append('<g>')
        self._stack.append(['</g>'])

    def pop(self):
        """ Pops the drawing state.
        """
        self.extend(reversed(self._stack.pop()))

    def translate(self, x, y):
        """ Moves the origin point by (x, y).
            By default, the origin point is the top left of the canvas.
        """
        self.append('<g transform="translate(%.1f, %.1f)">' % (x, y))
        self._stack[-1].append('</g>')

    def scale(self, x, y=None):
        """ Scales the transformation state (0.0-1.0).
        """
        self.append('<g transform="scale(%s, %s)">' % (x, x if y is None else y))
        self._stack[-1].append('</g>')

    def rotate(self, degrees):
        """ Rotates the transformation state.
        """
        self.append('<g transform="rotate(%.1f)">' % degrees)
        self._stack[-1].append('</g>')

    def background(self, *v, **k):
        """ Colors the canvas, e.g., background(1).
        """
        self._state['background'] = Color(*v, **k)

    def fill(self, *v, **k):
        """ Colors subsequent paths, rect(), ellipse(): fill(0, 0, 0, 0.5).
        """
        self.append('<g fill="%s">' % Color(*v, **k))
        self._stack[-1].append('</g>')

    def stroke(self, *v, **k):
        """ Colors subsequent outlines of paths, line(), rect(), ellipse().
        """
        self.append('<g stroke="%s">' % Color(*v, **k))
        self._stack[-1].append('</g>')

    def strokewidth(self, w):
        self.append('<g stroke-width="%.1f">' % w)
        self._stack[-1].append('</g>')

    def strokestyle(self, *w):
        self.append('<g stroke-dasharray="%s">' % ' '.join(map(str, w)))
        self._stack[-1].append('</g>')

    def nofill(self):
        self.append('<g fill="none">')
        self._stack[-1].append('</g>')

    def nostroke(self):
        self.append('<g stroke="none">')
        self._stack[-1].append('</g>')

    def color(self, *v, **k):
        return Color(*v, **k)

    def line(self, x1, y1, x2, y2, **k):
        """ Draws a line from x1, y1 to x2, y2: line(0, 0, 100, 100, stroke=color(0)).
        """
        s = '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" />' % (x1, y1, x2, y2)
        s = mixin(s, **k)
        self.append(s)

    def rect(self, x, y, w, h, roundness=0, **k):
        """ Draws a rectangle at x, y with width w and height h.
        """
        s = '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%.1f" />' % (x, y, w, h, roundness)
        s = mixin(s, **k)
        self.append(s)

    def ellipse(self, x, y, w, h, **k):
        """ Draws an ellipse at x, y with width w and height h.
        """
        s = '<ellipse cx="%.1f" cy="%.1f" rx="%.1f" ry="%.1f" />' % (x, y, w * 0.5, h * 0.5)
        s = mixin(s, **k)
        self.append(s)

    def arc(self, x, y, r, a1, a2, close=True, **k):
        """ Draws an arc at x, y with radius r from angle a1 to a2 (counterclockwise).
        """
        if abs(a2 - a1) >= 360:
            return self.ellipse(x, y, r * 2, r * 2, **k)
        a1, a2 = sorted((a1, a2))
        x1, y1 = coordinates(x, y, r, -a1)
        x2, y2 = coordinates(x, y, r, -a2)
        f = ('0', '1')[a2 - a1 >= 180]
        c = ('M', 'L')[close]
        s = '<path d="M %.1f %.1f A %.1f %.1f 0 %s 0 %.1f %.1f %s %.1f %.1f Z" />' % (x1, y1, r, r, f, x2, y2, c, x, y)
        s = mixin(s, **k)
        self.append(s)

    def beginpath(self, x, y):
        self._state['path'] = Path()
        self._state['path'].moveto(x, y)

    def moveto(self, x, y):
        self._state['path'].moveto(x, y)

    def lineto(self, x, y):
        self._state['path'].lineto(x, y)

    def curveto(self, ctrl1x, ctrl1y, ctrl2x, ctrl2y, x, y):
        self._state['path'].curveto(ctrl1x, ctrl1y, ctrl2x, ctrl2y, x, y)

    def closepath(self):
        self._state['path'].close()

    def endpath(self, draw=True, **k):
        p = self._state['path']
        if draw:
            self.drawpath(p, **k)
        return p

    def drawpath(self, path, winding=NONZERO, **k):
        s = '<path d="%s" fill-rule="%s" />' % (' '.join(map(str, path)), winding)
        s = mixin(s, **k)
        self.append(s)

    def findpath(self, points=[], curvature=0.5):
        return fit(points, curvature)

    def beginclip(self, path, winding=NONZERO):
        id = len(self._defs)
        p = '<path d="%s" clip-rule="%s" />' % (' '.join(map(str, path)), winding)
        p = '<clipPath id="%s">%s</clipPath>' % (id, p)
        s = '<g style="clip-path:url(#%s);">' % (id,)
        self._defs[id] = p
        self.append(s)
        self._stack.append(['</g>'])

    def endclip(self):
        self.extend(reversed(self._stack.pop()))

    def font(self, fontname, fontsize=None):
        if fontsize is None:
            fontsize = self._state['font'][1]
        self.append('<g font-family="%s" font-size="%ipx">' % (fontname, fontsize))
        self._stack[-1].append('</g>')
        self._state['font'][0] = fontname
        self._state['font'][1] = fontsize

    def fontsize(self, v):
        self.append('<g font-size="%ipx">' % v)
        self._stack[-1].append('</g>')
        self._state['font'][1] = v

    def fontweight(self, v):
        self.append('<g font-weight="%s">' % v)
        self._stack[-1].append('</g>')
        self._state['font'][2] = v

    def text(self, s, x, y, align=LEFT, **k):
        k.setdefault('strokewidth', 0)
        a = ('start', 'end', 'middle')[align]
        s = '<text x="%.1f" y="%.1f" text-anchor="%s">%s</text>' % (x, y, a, encode('%s' % s))
        s = mixin(s, **k)
        self.append(s)

    def textwidth(self, s, **k):
        return textsize(s,
            k.get('font'       , self._state['font'][0]),
            k.get('fontsize'   , self._state['font'][1]),
            k.get('fontweight' , self._state['font'][2]),
        )[0]

    def image(self, path, x, y, width=None, height=None, opacity=1.0, **k):
        """ Draws the given image file at x, y.
        """
        id = 'img-' + hashlib.md5(path.encode('utf-8')).hexdigest()[:16]

        # Cache image data (once):
        if not path in self._defs:
            w, h = self.imagesize(path)
            s = data_uri(path) # 'data:image/png;base64,...'
            s = '<image id="%s" width="%i" height="%i" href="%s" />' % (id, w, h, s)
            self._defs[id] = s
        w = self._defs[id].split(' ', 3)[2][7:-1]
        h = self._defs[id].split(' ', 4)[3][8:-1]
        w = float(w)
        h = float(h)

        def pct(v1, v2):
            return float(v1) / v2 if v2 != 0 else 0.0
        try:
            w = pct(width, w)
        except:
            w = pct(height or h, h)
        try:
            h = pct(height, h)
        except:
            h = w

        s = '<use href="#%s" x="%.1f" y="%.1f" style="opacity:%.2f;" />' % (id, x, y, opacity)
        s = mixin(s, **k)
        self.append('<g transform="scale(%s, %s)">' % (w, h))
        self.append(s)
        self.append('</g>')

    def imagesize(self, path):
        """ Returns the (width, height) of the given image file.
        """
        w = 0
        h = 0
        f = open(path, 'rb')
        s = f.read(32)
        if path.endswith('.gif'):
            w, h = struct.unpack('<HH', s[ 6:10])
        if path.endswith('.png'):
            w, h = struct.unpack('>LL', s[ 8:16]) if s[12:16] != b'IHDR' else \
                   struct.unpack('>LL', s[16:24]) 
        if path.endswith('.jpg'):
            f.seek(0)
            s = f.read(2)
            s = f.read(1)
            while s != b'\xda':
                while s != b'\xff': 
                    s = f.read(1)
                while s == b'\xff': 
                    s = f.read(1)
                if s in (b'\xc0', b'\xc1', b'\xc2', b'\xc3'):
                    s = f.read(3)
                    s = f.read(4)
                    h, w = struct.unpack('>HH', s)
                    break
                n,= struct.unpack('>H', f.read(2))
                s = f.read(int(n) - 2)
                s = f.read(1)
        return int(w), int(h)

    def random(self, *v):
        if len(v) == 0:    # random()
            a, b = 0, 1
        if len(v) == 1:    # random(1.0)
            a, b = 0, v[0]
        if len(v) == 2:    # random(0.0, 1.0)
            a, b = v
        x = _random.random()
        x = x * (b - a) + a

        if isinstance(a, int) and \
           isinstance(b, int):
            return int(x)
        else:
            return x

    def __repr__(self):
        return 'Context(width=%.1f, height=%.1f)' % (
            self.width, 
            self.height
        ) 

    def __unicode__(self):
        w  = self.width
        h  = self.height
        s  = u''
        s += '<svg version="1.2" width="%i" height="%i" xmlns="http://www.w3.org/2000/svg">\n' % (w, h)
        s += '<style>a { text-decoration: underline; }</style>'
        s += '<defs>\n' 
        s += '\n'.join(self._defs.values())
        s += '\n'
        s += '</defs>\n'
        s += '<g fill="rgba(0,0,0,1)" stroke="none" stroke-width="1">\n'
        s += '<g font-family="Arial" font-size="12px">\n'
        s += '<rect width="%i" height="%i" fill="%s" />\n' % (w, h, self._state['background'])
        s += '\n'.join(self)
        s += '\n'
        s += '\n'.join('\n'.join(reversed(g)) for g in self._stack)
        s += '\n'
        s += '</g>\n'
        s += '</g>\n'
        s += '</svg>'
        s  = s.replace('<defs>\n\n</defs>', '', 1)
        s  = s.replace('\n\n', '\n')
        return s

    def __str__(self):
        if PY2:
            return self.__unicode__().encode('utf-8')
        else:
            return self.__unicode__()

    def render(self):
        return self.__unicode__()

    def save(self, path):
        with codecs.open(path, 'w', encoding='utf-8') as f:
            f.write(XML + '\n' + self.__unicode__())

    def clear(self):
        del self[:]

#--------------------------------------------------------------------------------------------------
# Initialize a global Context _ctx and expose its methods as functions: rect() = _ctx.rect().

_ctx = Context(_global=True)

for f in dir(_ctx):
    if f.startswith('_'):
        continue
    if f in ('width', 'height'):
        continue
    if f != 'pop' and hasattr(list, f):
        continue
    globals()[f] = getattr(_ctx, f)
