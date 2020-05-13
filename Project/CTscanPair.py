import io
from PIL import Image
from PIL import ImageFilter
from pylab import *

class CTscanPair:
    # Class attributes

    # Scikit- image

    # OpenCV image, attention to conversion RGB - BGR

    # PIL image

    # Constructor / instances attributes
    #   Pair: tuple with two paths to the preop and postop CT scan images.
    def __init__(self, pair):
        self.prePath = pair[1]
        self.postPath = pair[0]

        with open(self.prePath, 'rb') as f:
            self.prePIL = array(Image.open(io.BytesIO(f.read())).convert('L'))

        with open(self.postPath, 'rb') as g:
            self.postPIL = array(Image.open(io.BytesIO(g.read())).convert('L'))




    def getPreImg(self):
        return self.prePIL

    def getPostImg(self):
        return self.postPIL


    def getSpiralCenter(self):
        im = self.prePIL
        #im1 = im.filter(ImageFilter.GaussianBlur)

        return im
