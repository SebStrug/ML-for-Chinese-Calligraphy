# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:34:39 2017

@author: Sebastian
"""
import PIL
from PIL import Image, ImageDraw, ImageFont

s = u'\u3400'
with open("yop.txt", "wb") as f:
   f.write(s.encode("UTF-8"))

setSize = 300
unicode_text = u"\u3500"
simsum_font = ImageFont.truetype('simsun.ttc',setSize)
#font must be supported

img = Image.new('RGB', (setSize, setSize), (255,255,255))
draw = ImageDraw.Draw(img) 
draw.text((1,1), unicode_text, font = simsum_font, fill = "#000000")
#img.save('out.png')
img.save('withBlur.png')
newImg = img.convert('L') #greyscale the image
newImg1 = newImg.point(lambda x: 0 if x<120 else 255, '1') #set threshold for greyscale to binarize the image
newImg2 = newImg.point(lambda x: 0 if x<140 else 255, '1') #set threshold for greyscale to binarize the image
newImg3 = newImg.point(lambda x: 0 if x<160 else 255, '1') #set threshold for greyscale to binarize the image
newImg1.save('noBlur1.png')
newImg2.save('noBlur2.png')
newImg3.save('noBlur3.png')