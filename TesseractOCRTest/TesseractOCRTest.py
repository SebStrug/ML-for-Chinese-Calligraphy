"""
Created on Tue Sep 26 13:48:26 2017
Testing Tesseract-OCR
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

print(pytesseract.image_to_string(Image.open('test-eng.png'), lang='eng',config=tessdata_dir_config))
print(pytesseract.image_to_string(Image.open('test-fonts.png'), lang='eng',config=tessdata_dir_config))
print(pytesseract.image_to_string(Image.open('test-european.jpg'), lang='fra+ita+spa+por', config=tessdata_dir_config))

chiCaritas = pytesseract.image_to_string(Image.open('caritas.png'), lang='chi_sim+chi_tra', config=tessdata_dir_config)
print(chiCaritas)
engTranslation = mt.translate(chiCaritas.encode('utf-8'),"en","auto")
print(engTranslation)