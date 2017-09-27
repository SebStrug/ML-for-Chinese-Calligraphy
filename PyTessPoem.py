# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:48:26 2017

@author: Sebastian
"""
try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract

import PIL
import mtranslate as mt

pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'
# Include the above line, if you don't have tesseract executable in your PATH
# Example tesseract_cmd: 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'
#
#print(pytesseract.image_to_string(Image.open('test.png')))
#
tessdata_dir_config = '--tessdata-dir "C:\Program Files (x86)\Tesseract-OCR"'
# Example config: '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
# It's important to add double quotes around the dir path.
#
#print(pytesseract.image_to_string(Image.open('test-european.jpg'), lang='fra+ita+spa+por', config=tessdata_dir_config))
#chiCode = pytesseract.image_to_string(Image.open('caritas.png'), lang='chi_sim+chi_tra', config=tessdata_dir_config)
#print(chiCode)
#
#line_en = mt.translate(chiCode.encode('utf-8'),"en","auto")
#print(line_en)

baotuImage = Image.open("BaotuSpringTest.jpg")
baotuImage.save("test-600.png", dpi=(300,300) )
baotuImageGrey2 = baotuImage.convert('L')
baotuImageGrey2 = baotuImageGrey2.point(lambda x: 0 if x<100 else 255, '1')
baotuImageGrey2.save("BaotuSpringGrey2.png")

chiCode2 = pytesseract.image_to_string(Image.open('BaotuSpringGrey2.png'),lang='eng+chi_sim+chi_trad',config=tessdata_dir_config)
print(chiCode2)
print(mt.translate(chiCode2.encode('utf-8'),"en","auto"))