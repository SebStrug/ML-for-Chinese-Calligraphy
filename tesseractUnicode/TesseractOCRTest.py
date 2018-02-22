"""
Created on Tue Sep 26 13:48:26 2017
Testing Tesseract-OCR
@author: Sebastian, Elliot
"""
import os

try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract #python wrapper for Tesseract-OCR
import mtranslate as mt #translater for Chinese -> English

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract' #change Tesseract-OCR path location
tessdata_dir_config = '--tessdata-dir "C:\Program Files (x86)\Tesseract-OCR"' #path for Tesseract-OCR language data

#%%
path = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Tesseract and Unicode\\TesseractOCRTest'
os.chdir(path)

##%%
imagePath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Tesseract and Unicode\\TesseractOCRTest\\dgr_file.png'

#%%
text = pytesseract.image_to_string(Image.open(imagePath), lang='chi_sim+chi_tra', config=tessdata_dir_config)
print(text)
engTranslation = mt.translate(text.encode('utf-8'),"en","auto")
print(engTranslation)
 