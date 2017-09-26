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

pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'
# Include the above line, if you don't have tesseract executable in your PATH
# Example tesseract_cmd: 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'

print(pytesseract.image_to_string(Image.open('test.png')))

tessdata_dir_config = '--tessdata-dir "C:\Program Files (x86)\Tesseract-OCR"'
# Example config: '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
# It's important to add double quotes around the dir path.

print(pytesseract.image_to_string(Image.open('test-european.jpg'), lang='fra+ita+spa+por', config=tessdata_dir_config))

#print(pytesseract.image_to_string(Image.open('test-european.jpg'), lang='fra'))
#comment
