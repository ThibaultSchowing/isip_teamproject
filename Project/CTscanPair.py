import io
from PIL import Image
import utils
from PIL import ImageFilter
from pylab import *
import cv2


# BASE IMAGE ALWAYS IN NPARRAY FORMAT with values 0-255 ! (it's easier to handle)
class CTscanPair:
    # Class attributes

    # Constructor / instances attributes
    #   Pair: tuple with two paths to the preop and postop CT scan images.
    def __init__(self, pair, patternpaths):
        self.prePath = pair[1]
        self.postPath = pair[0]
        self.patternPaths = patternpaths
        self.pattern_arr = []

        with open(self.prePath, 'rb') as f:
            self.preop_arr = array(Image.open(io.BytesIO(f.read())).convert('L'))

        with open(self.postPath, 'rb') as g:
            self.postop_arr = array(Image.open(io.BytesIO(g.read())).convert('L'))

        # List of pattern (orientation of inner ear), binary 0-255. Bitwisenot is here to reverse the wrongly made picture
        for patt in self.patternPaths:
            with open(patt, 'rb') as p:
                self.pattern_arr.append(cv2.bitwise_not(array(Image.open(io.BytesIO(p.read())).convert('L'))))

    def getPreImg(self):
        return self.preop_arr

    def getPostImg(self):
        return self.postop_arr

    # Idea: use template matching (CV2) or normalized cross correlation to find the/a spiral in the picture.
    def getSpiralCenter(self):
        im = self.preop_arr.copy()

        # Preprocess image (gaussian filter / smoothing)

        # Apply threshold for grey values
        # Threshold chosen according to image histogram showing peak between 50 and 100 for grey
        # values.

        # Noise reduction
        im = cv2.GaussianBlur(im, (5,5),15)

        img = np.where(np.logical_or(im > 100, im < 50), 255, 0).astype(np.uint8)
        utils.show(img, "threshold")

        im_canny = cv2.Canny(img, 5, 5)
        utils.show(im_canny, "canny edge detect")

        # Preprocess pattern
        pattern = cv2.Canny(self.pattern_arr[1],5 ,5)
        utils.show(pattern, "pattern")

        center_list, methods, score_list = utils.best_match_location(im_canny, pattern)


        for i in range(0,len(center_list)):
            cv2.circle(img, center_list[i], 30, 150, 10)
            plt.subplot(121), plt.imshow(score_list[i], cmap='gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(img, cmap='gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.suptitle(methods[i] + self.prePath)


        return 66
