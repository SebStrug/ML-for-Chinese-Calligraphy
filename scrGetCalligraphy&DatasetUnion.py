# -*- coding: utf-8 -*-
"""
Created on Tue May  1 12:15:44 2018

@author: ellio
"""
import os 
gitHubRep=os.getcwd()

#os.chdir(os.path.join(gitHubRep,"dataHandling/"))
#from classFileFunctions import fileFunc as fF 

import numpy as np

charListPath = gitHubRep
calligCharListPath = gitHubRep
charListName="List_of_chars_NUMPY.npy"
calligCharListName = "calligCharList.txt"

#load character list for CASIA dataset
os.chdir(charListPath)
with open(charListName, 'rb') as ifs:
    fileNPZ = np.load(ifs)
charListCASIA = fileNPZ[171:]

#load Calligraphy character list
calligChars="行书\
趵突泉\
泺水发源天下无，平\
地涌出白玉壶。谷虚\
久恐元气泄，岁旱不\
愁东海枯。云雾润\
蒸华不注，波澜声\
震大明湖。时来泉上\
濯尘土，冰雪满怀清\
兴孤。\
\
行书\
右二题皆济南近\
郭佳处，公谨家故\
齐也，遂為书此。孟頫。\
\
行书\
右追咏松雪道人\
趵突泉诗一首\
长干沙门守仁书\
时洪武己巳春三\
月十日也\
\
行书\
趵突奔泉世所无，历城佳\
处似蓬壶。源头但欠千波\
涌，海眼何曾一日枯。平地\
风雷藏尺泽，满天星斗\
落澄湖。济南名士多如\
雨，翰苑题诗兴不孤。\
富春如兰\
\
楷书\
魏国风流一代贤，才华如\
锦笔如椽。为官昔在济\
南郡，出郭曾题趵突泉。\
鹄峙鸾停文力健，龙蟠\
凤翥墨花鲜。玉堂学士\
今何有，开卷令人独怆然。 \
吴沙门弘道\
\
行书\
玉堂文采古今无，历下\
亭前酒一壶。弹詈\
欲同丘岳重，墨痕\
不逐水泉枯。谪仙\
壮气横青海，贺老\
清风满鉴湖。春日题\
诗重怅望，济南\
山远白云孤。\
\
楷书\
松雪老人词翰妙绝天下，当元初至元大德间，馆\
阁诸公皆推尊之。下至闾巷小儿，莫不知其\
姓名。非其德行之重、材学之美有以震耀乎？当\
时能若尔乎？或谓元朝士大夫字画声诗\
之盛，一变夫故宋余习，（盖自公始），信然。今观其所书趵\
突泉诗，令人叹赏不已。宁上人得此，其善\
保之。洪武己巳春三月 天台沙门清浚识。\
"


#
#os.chdir(calligCharListPath)
#with open(calligCharListName,'rb') as ifs:
#   byteList=ifs.read()
#chars=[]
#i=0
#while(i<len(byteList)):
#    chars.append(byteList[i:i+2].decode(encoding='gbk'))
#    i+=2
calligCharList = set(calligChars)
intersectionCharList =calligCharList.intersection(set(charListCASIA)  )

    