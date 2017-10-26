# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:40:14 2017

@author: Sebastian
"""
import PIL
from PIL import Image, ImageDraw, ImageFont
import os, glob
import time
import pytesseract #python wrapper for Tesseract-OCR
import mtranslate as mt #translater for Chinese -> English

setSize = 2000
borderSize = 200
howMany = 30 #how many characters to do

#now use the CASIA 1.0 database to do this for their 3740 characters
with open("Chinese3740-list.txt", "rb") as Chinese3740:
    Chinese3740.readline(); Chinese3740.readline(); #ignore headers
    content = [x.strip() for x in Chinese3740.readlines()] #place lines into separate arrays
    unicodeEntries = [i.split()[2] for i in content] #only look at the unicode entry for each character
    unicodeEntries = [chr(int(i,16)) for i in unicodeEntries]

k = 0 #make variable to keep track
cwd = os.getcwd() #get path of current working directory
subDir = os.path.join(cwd, 'Chinese30Characters')
for i in range(howMany):
    img = Image.new('RGB', ((setSize+borderSize), int((setSize+borderSize)/10)), (255,255,255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('simsun.ttc',int(setSize/10)) #simsum.ttc
    draw.text((int((borderSize)/2), 0), ' '.join(unicodeEntries[i]), font = font, fill = "#000000") #simsum_font
    print(i)
    img = img.convert('L') #greyscale the image
    img = img.point(lambda x: 0 if x<180 else 255, '1') #set threshold for greyscale to bina
    imageSavePath = os.path.join(subDir, 'Char {}.png'.format(k)) #i or k depending on number or character)
    img.save(imageSavePath)
    k += 1
del k #clear k variable

pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract' #change Tesseract-OCR path location
tessdata_dir_config = '--tessdata-dir "C:\Program Files (x86)\Tesseract-OCR"' #path for Tesseract-OCR language data
#calligOCR = pytesseract.image_to_string(Image.open('Char 1.png'),lang='eng+chi_sim+chi_tra',config=tessdata_dir_config)
#print(calligOCR)



#os.chdir("C://Users//Sebastian//Desktop//MLChinese//unicodeTest//Chinese30Characters") #change directory
#charList = [val[5] for val in glob.glob("*.png")] #append all characters we have as image files into an array


tic = time.clock()
rightVals = 0
wrongVals = 0
for i in range(0,howMany):
    imagePath = os.path.join(subDir, 'Char {}.png'.format(i)) #charList[i-1] instead of k
    calligOCR = pytesseract.image_to_string(Image.open(imagePath),lang='eng+chi_sim+chi_tra',config=tessdata_dir_config)
    if calligOCR == '': 
        calligOCR = 'blank'
    
    if calligOCR != unicodeEntries[i]:
        wrongVals += 1
        print(calligOCR, unicodeEntries[i], 'wrong', i)
    else: 
        rightVals += 1
        print(calligOCR, unicodeEntries[i], 'right', i)
#    #print(mt.translate(calligOCR.encode('utf-8'),"en","auto"))
toc = time.clock()
print(toc-tic, rightVals+wrongVals)
print(rightVals, wrongVals)
    
#run tesseract on these images and see how many are correctly recognised 