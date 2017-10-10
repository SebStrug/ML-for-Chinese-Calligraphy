# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 21:10:13 2017
Trying to use Tesseract-OCR on Chinese Calligraphy
@author: Sebastian, Elliot
"""

try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract #python wrapper for Tesseract-OCR
import mtranslate as mt #translater for Chinese -> English

pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract' #change Tesseract-OCR path location
tessdata_dir_config = '--tessdata-dir "C:\Program Files (x86)\Tesseract-OCR"' #path for Tesseract-OCR language data

#baotuImage = Image.open("Char 3.png") #test from char images
baotuImage = Image.open("BaotuSpringTest.jpg") #open test image of calligraphy
baotuImage.save("test-600.png", dpi=(300,300) ) #change dpi to 300, does this make a difference?
baotuImageGrey = baotuImage.convert('L') #greyscale the image
baotuImageGrey = baotuImageGrey.point(lambda x: 0 if x<100 else 255, '1') #set threshold for greyscale to binarize the image
baotuImageGrey.save("BaotuSpringGrey.png") #save image

calligOCR = pytesseract.image_to_string(Image.open('BaotuSpringGrey.png'),lang='eng+chi_sim+chi_trad',config=tessdata_dir_config)
print(calligOCR)
print(mt.translate(calligOCR.encode('utf-8'),"en","auto"))