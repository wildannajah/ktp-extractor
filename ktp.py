import PIL
from PIL import ImageDraw
import sys
import os
import re
import math
import copy
import pandas as pd
import numpy as np
import bisect
from datetime import datetime
from dateutil.parser import parse

fields_ktp = [
    {'field_name': 'provinsi', 'keywords': 'provinsi', 'typo_tolerance': 4},
    {'field_name': 'kota', 'keywords': 'kabupaten', 'typo_tolerance': 1},
    {'field_name': 'nik', 'keywords': 'nik', 'typo_tolerance': 4},
    {'field_name': 'nama', 'keywords': 'nama', 'typo_tolerance': 4},
    {'field_name': 'ttl', 'keywords': 'tempat/tgl', 'typo_tolerance': 4},
    {'field_name': 'jenis_kelamin', 'keywords': 'kelamin', 'typo_tolerance': 4},
    {'field_name': 'gol_darah', 'keywords': 'darah:', 'typo_tolerance': 1},
    {'field_name': 'alamat', 'keywords': 'alamat', 'typo_tolerance': 4},
    {'field_name': 'rt_rw', 'keywords': 'rt/rw', 'typo_tolerance': 4},
    {'field_name': 'kel_desa', 'keywords': 'kel/desa', 'typo_tolerance': 4},
    {'field_name': 'kecamatan', 'keywords': 'kecamatan', 'typo_tolerance': 4},
    {'field_name': 'agama', 'keywords': 'agama', 'typo_tolerance': 4},
    {'field_name': 'status_perkawinan', 'keywords': 'perkawinan', 'typo_tolerance': 1},
    {'field_name': 'pekerjaan', 'keywords': 'pekerjaan', 'typo_tolerance': 4},
    {'field_name': 'kewarganegaraan', 'keywords': 'kewarganegaraan', 'typo_tolerance': 4},
    {'field_name': 'berlaku_hingga', 'keywords': 'berlaku', 'typo_tolerance': 2}
]

list_awalan_provinsi = ["provinsi sumatra","provinsi dki jakarta","provinsi jawa", "provinsi papua", "provinsi kalimantan", "provinsi nusa","provinsi nusa tenggara", "provinsi maluku","provinsi sumatera", "provinsi bali","provinsi aceh", "provinsi daerah","provinsi banten","provinsi bengkulu","provinsi bali", "provinsi gorontalo","provinsi jambi","provinsi kepulauan"]
mata_angin = ["utara", "selatan", "timur", "tenggara", "barat"]

alter_negara = ["kewarga", "negaraan","kewarganegaraan wni"]
alter_berlaku = ["berlaku", "berlaku hingga","berlaku hingga seumur","berlaku hingga seumur hidup"]
alter_ttl = ["tempattgl lahir", "tempattgl"]
alter_kawin = ["kawin", "belum kawin", "tidak kawin", "janda", "duda", "cerai"]
alter_gol_darah = ['A','B','AB','O']

def easyocr_format_to_gcv_format(output_easyocr):
    # """hasil dari easyocr dibuat format google cloud vision mengikuti repo bukalapak"""
  list_output = []
  for bound in output_easyocr:
    dict_data = {}
    for i,bb in enumerate(bound[0]):
      dict_data[f"x{i+1}"],dict_data[f"y{i+1}"] = bb[0],bb[1]
    dict_data["h"] = int(abs(dict_data["x2"] - dict_data["x1"]))
    dict_data["w"] = int(abs(dict_data["y2"] - dict_data["y1"]))
    dict_data["label"] = bound[1]
    list_output.append(dict_data)

  return list_output

def levenshtein(source, target):
    """menghitung jarak levenshtein dari source ke target"""
    if len(source) < len(target):
        return levenshtein(target, source)

    if len(target) == 0:
        return len(source)

    source = np.array(tuple(source))
    target = np.array(tuple(target))

    previous_row = np.arange(target.size + 1)
    for s in source:
        current_row = previous_row + 1
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]

def correct2numbers(words):
    words = words.replace(' ','')
    if isNumber(words):
        result = ''
        for cc in words:
            if cc in ['T','I']:
                result+='1'
            elif cc>='0' and cc<='9' :
                result+=cc
            else :
                result+='0'
        words = result
    return words

def calDegBox(box,x,y,w):
    """menghitung sudut dari 2 box"""
    ls_cal_abs = [np.abs(nx-x)+np.abs(ny-y) for nx,ny in box]
    index = np.argmin(ls_cal_abs)

    ls_cal_abs2 = [np.abs(nx-x-w)+np.abs(ny-y) for nx,ny in box]
    index2 = np.argmin(ls_cal_abs2)

    x1,y1 = box[index]
    x2,y2 = box[index2]
    myradians = math.atan2(y1-y2, x1-x2)
    mydegrees = math.degrees(myradians)
    mydegrees = mydegrees if mydegrees >= 0 else 360+mydegrees
    return mydegrees

def calDeg(x1,y1,x2,y2):
    myradians = math.atan2(y1-y2, x1-x2)
    mydegrees = math.degrees(myradians)
    mydegrees = mydegrees if mydegrees >= 0 else 360+mydegrees
    return mydegrees

def convert_format(text_response):
    ls_word = []
    if ('textAnnotations' in text_response):
        for text in text_response['textAnnotations']:
            boxes = {}
            boxes['label'] = text['description']

            boxes['x1'] = text['boundingPoly']['vertices'][0].get('x',0)
            boxes['y1'] = text['boundingPoly']['vertices'][0].get('y',0)
            boxes['x2'] = text['boundingPoly']['vertices'][1].get('x',0)
            boxes['y2'] = text['boundingPoly']['vertices'][1].get('y',0)
            boxes['x3'] = text['boundingPoly']['vertices'][2].get('x',0)
            boxes['y3'] = text['boundingPoly']['vertices'][2].get('y',0)
            boxes['x4'] = text['boundingPoly']['vertices'][3].get('x',0)
            boxes['y4'] = text['boundingPoly']['vertices'][3].get('y',0)

            boxes['w'] = boxes['x3'] - boxes['x1']
            boxes['h'] = boxes['y3'] - boxes['y1']

            ls_word.append(boxes)
    return ls_word

def get_attribute_ktp(ls_word,field_name,field_keywords,typo_tolerance, debug_mode=False):
    """mendapatkan attribut dari foto ktp"""
    if(len(ls_word)==0):
        return None

    if(field_name == 'nama'):
        ls_word = np.asarray([word for word in ls_word if word['label'].lower() not in ['jawa','nusa'] ])

    
    print(f'field keywords  : {field_keywords}')

    new_ls_word = np.asarray([word['label'].lower() for word in ls_word])
    print(f'new_ls_word  : {new_ls_word}')
    
    for word in new_ls_word:
        print(f'word : {word}')

    ls_dist = [levenshtein(field_keywords, word.lower()) for word in new_ls_word]
    print(f'ls_dist awal : {ls_dist}')
    print(f'np min ls_dist awal : {np.min(ls_dist)}')
    
    provinsi_search = False
    provinsi_search_angin = False

    if(np.min(ls_dist) > typo_tolerance and field_name=="provinsi"):
        for awalan_provinsi in list_awalan_provinsi:
            ls_dist = [levenshtein(awalan_provinsi, word.lower()) for word in new_ls_word]
            if np.min(ls_dist) <= typo_tolerance:
                provinsi_search = True
                ls_dist = ls_dist
                break
            else:
                continue
    
    # handle mata angin di provinsi
    if(np.min(ls_dist) > typo_tolerance and field_name=="provinsi"):
        for angin in mata_angin:
            for awalan_provinsi in list_awalan_provinsi:
                prov = awalan_provinsi+" " +angin
                print(prov)
                print(word.lower() for word in new_ls_word)
                ls_dist_alter = [levenshtein(prov, word.lower()) for word in new_ls_word]
                print(f'ls_dist alter: {ls_dist_alter}')
                print(f'np_min_ls_dist: {int(np.min(ls_dist_alter))}')
                print(f'typo_tolerance: {typo_tolerance}')
                if np.min(ls_dist_alter) <= typo_tolerance:
                    provinsi_search_angin = True
                    print(f'ls_dist dalam else: {ls_dist}')
                    ls_dist = ls_dist_alter
                    print(f'ls_dist dalam else alter: {ls_dist}')
                    break
                else:
                    continue

    # handle word yang nyambung misal ("perkawinan : belum kawin")
    if(np.min(ls_dist) > typo_tolerance and field_name=="status_perkawinan"):
        for kawin in alter_kawin:
            key_kawin = "perkawinan:"+" " + kawin
            print(key_kawin)
            ls_dist_alter = [levenshtein(key_kawin, word.lower()) for word in new_ls_word]
            print(f'ls_dist alter: {ls_dist_alter}')
            print(f'np_min_ls_dist: {int(np.min(ls_dist_alter))}')
            print(f'typo_tolerance: {typo_tolerance}')
            if np.min(ls_dist_alter) <= typo_tolerance:
                field_value = kawin
                return field_value
            else:
                continue

    # handle gol_darah with low typo_tolerance
    if(np.min(ls_dist) > typo_tolerance and field_name=="gol_darah"):
        gol = ["darah", "gol darah" ,"darah:", "darah :", "gol.darah:", "gol.darah :"]
        for goldar in alter_gol_darah:
            for g in gol:
                print(g)
                ls_dist_alter = [levenshtein(g, word.lower()) for word in new_ls_word]
                print(f'ls_dist alter tes: {ls_dist_alter}')
                print(f'np_min_ls_dist: {int(np.min(ls_dist_alter))}')
                print(f'typo_tolerance: {typo_tolerance}')
                if np.min(ls_dist_alter) < typo_tolerance:
                    ls_dist = ls_dist_alter
                    print(f'ls_dist dalam else alter  goldar: {ls_dist}')
                    break
                else : 
                    key_goldar = g + " " + goldar.lower()
                    print(key_goldar)
                    ls_dist_alter = [levenshtein(key_goldar, word.lower()) for word in new_ls_word]
                    print(f'ls_dist alter tes: {ls_dist_alter}')
                    print(f'np_min_ls_dist: {int(np.min(ls_dist_alter))}')
                    print(f'typo_tolerance: {typo_tolerance}')
                    if np.min(ls_dist_alter) < typo_tolerance:
                        field_value = goldar
                        return field_value
            else:
                continue

    kota_search = False
    jakarta_search = False
    # handle gol_darah with low typo_tolerance
    if(np.min(ls_dist) > typo_tolerance and field_name=="kota"):
        alter_kota = ["kota", "jakarta"]
        for goldar in mata_angin:
            for g in alter_kota:
                print(f"g : {g}")
                if g == "kota":
                    kota_search =  True
                else: 
                    jakarta_search = True
                    kota_search = False
                    print(f"kota search = {kota_search}")
                    print(f"jakarta search = {jakarta_search}")

                ls_dist_alter = [levenshtein(g, word.lower()) for word in new_ls_word]
                print(f'ls_dist alter tes: {ls_dist_alter}')
                print(f'np_min_ls_dist: {int(np.min(ls_dist_alter))}')
                print(f'typo_tolerance: {typo_tolerance}')
                if np.min(ls_dist_alter) < typo_tolerance:
                    ls_dist = ls_dist_alter
                    print(f'ls_dist dalam else alter  goldar: {ls_dist}')
                    break
                else : 
                    key_goldar = g + " " + goldar.lower()
                    print(key_goldar)
                    ls_dist_alter = [levenshtein(key_goldar, word.lower()) for word in new_ls_word]
                    print(f'ls_dist alter tes: {ls_dist_alter}')
                    print(f'np_min_ls_dist: {int(np.min(ls_dist_alter))}')
                    print(f'typo_tolerance: {typo_tolerance}')
                    if np.min(ls_dist_alter) < typo_tolerance:
                        field_value = goldar
                        return field_value
            else:
                continue


    if(np.min(ls_dist) > typo_tolerance and field_name=="ttl"):
        for alter in alter_ttl:
            ls_dist = [levenshtein(alter, word.lower()) for word in new_ls_word]
            if np.min(ls_dist) > typo_tolerance:
                continue
            else:
                if len(ls_dist) < 1:
                  return None
                ls_dist = ls_dist
                break

# alter_negara = ["kewarga", "negaraan","kewarganegaraan wni"]
    if(np.min(ls_dist) > typo_tolerance and field_name=="kewarganegaraan"):
        for alter in alter_negara:
            ls_dist = [levenshtein(alter, word.lower()) for word in new_ls_word]
            if np.min(ls_dist) > typo_tolerance:
                continue
            elif(np.min(ls_dist) < typo_tolerance and alter == alter_negara[2]):
                field_value = "WNI"
                return field_value
            else:
                if len(ls_dist) < 1:
                  return None
                ls_dist = ls_dist
                break

            
# alter_kawin = ["kawin", "belum", "tidak", "janda", "duda", "cerai"]
    if(np.min(ls_dist) > typo_tolerance and field_name=="status_perkawinan"):
        print("ini perkawinan")
        for alter in alter_kawin:
            ls_dist = [levenshtein(alter, word.lower()) for word in new_ls_word]
            if np.min(ls_dist) > typo_tolerance:
                continue
            elif(np.min(ls_dist) < typo_tolerance and alter == alter_kawin[0]):
                field_value = "KAWIN"
                return field_value
            elif(np.min(ls_dist) < typo_tolerance and alter == alter_kawin[1]):
                field_value = "BELUM KAWIN"
                return field_value
            elif(np.min(ls_dist) < typo_tolerance and alter == alter_kawin[2]):
                field_value = "TIDAK KAWIN"
                return field_value
            elif(np.min(ls_dist) < typo_tolerance and alter == alter_kawin[3]):
                field_value = "JANDA"
                return field_value
            elif(np.min(ls_dist) < typo_tolerance and alter == alter_kawin[4]):
                field_value = "DUDA"
                return field_value
            elif(np.min(ls_dist) < typo_tolerance and alter == alter_kawin[5]):
                field_value = "CERAI"
                return field_value
            else:
                if len(ls_dist) < 1:
                  return None
                ls_dist = ls_dist
                break

        
        
# alter_berlaku = ["berlaku", "berlaku hingga","berlaku hingga seumur","berlaku hingga seumur hidup"]        
    if(np.min(ls_dist) > typo_tolerance and field_name=="berlaku_hingga"):        
        for alter in alter_berlaku:
            ls_dist = [levenshtein(alter, word.lower()) for word in new_ls_word]
            if np.min(ls_dist) > typo_tolerance:
                continue
            elif(np.min(ls_dist) < typo_tolerance and alter == alter_berlaku[2]):
                field_value = "SEUMUR HIDUP"
                return field_value
            elif(np.min(ls_dist) < typo_tolerance and alter == alter_berlaku[3]):
                field_value = "SEUMUR HIDUP"
                return field_value
            else:
                if len(ls_dist) < 1:
                  return None
                ls_dist = ls_dist
                break
        
    # if np.min(ls_dist) > typo_tolerance:
    #     if(field_name == 'kota' and field_keywords!='kota'):
    #         return get_attribute_ktp(ls_word,field_name,'kota',1,debug_mode)
    #     return None

    if(jakarta_search):
        print("handle index kota jakarta")
        index = np.argmin(ls_dist) + 1
    else :
        print("handle index kota")
        index = np.argmin(ls_dist)
    # index = np.argmin(ls_dist)
    print(f'index : {index}')
    x,y = ls_word[index]['x1'], ls_word[index]['y1']
    print(f'x : {x}')
    print(f'y : {y}')
    www = ls_word[index]
    print(f'word dari index : {www}')
    w = ls_word[index]['w']
    print(f'w : {w}')
    degree = calDeg(ls_word[index]['x1'],ls_word[index]['y1'],ls_word[index]['x2'],ls_word[index]['y2'])
    print(f'degree : {degree}')
    ls_y = np.asarray([np.abs(y-word['y1'])<300 for word in ls_word])
    print(f'ls_y : {ls_y}')

    value_words = [ww for ww, val in zip(ls_word,ls_y) if (val and np.abs(calDeg(x,y,ww['x1'],ww['y1'])-degree)<3)]
    print(f'value_words 1 : {value_words}')

    value_words_new = ls_word[index]["label"]
    print(f'value_words_new : {value_words_new}')

    if debug_mode:
        print(value_words)

    # handling special attributes

    value_words = [val for val in value_words if len(val['label'].replace(' ','').replace(':',''))>0]
    print(f'value_words 2: {value_words}')


    if(field_name=="provinsi"):
        pass
    else:
        d = [levenshtein('gol.', str(val['label']).lower()) for val in value_words]
        if(len(d)>0 and min(d) <= 1):
            idx = np.argmin(d)
            value_words.pop(idx)

        d = [levenshtein('darah', str(val['label']).lower()) for val in value_words]
        if(len(d)>0 and min(d) <= 1):
            idx = np.argmin(d)
            value_words.pop(idx)
        
        d = [levenshtein('gol. darah', str(val['label']).lower()) for val in value_words]
        if(len(d)>0 and min(d) <= 1):
            idx = np.argmin(d)
            value_words.pop(idx)

    if(field_name == 'nik'):
        if(len(value_words)>0):
            global max_x
            max_x = max([val['x2'] for val in value_words])

    if(field_name == 'kota'):
        field_value = ""
        for val in value_words:
            field_value = field_value + ' '+ str(val['label'])
        field_value = field_value.lstrip()

        # if(field_keywords == 'kabupaten'):
        #     return 'KABUPATEN '+field_value
        # else:
        #     return 'KOTA '+field_value
        if(kota_search):
            return 'KOTA '+field_value
        elif(jakarta_search):
            return 'JAKARTA '+field_value
        else:
            return 'KABUPATEN '+field_value

    if(field_name == 'ttl'):
            d = [levenshtein('lahir', str(val['label']).lower()) for val in value_words]
            if(len(d)>0 and min(d) <= 2):
                idx = np.argmin(d)
                value_words.pop(idx)

    elif(field_name == 'jenis_kelamin'):
            score_laki, score_wanita = 999,999
            d = [levenshtein('laki-laki', str(val['label']).lower()) for val in value_words]
            if(len(d)>0 and min(d) <= 2):
                return 'LAKI-LAKI'

            d = [levenshtein('laki', str(val['label']).lower()) for val in value_words]
            if(len(d)>0 and min(d) <= 1):
                return 'LAKI-LAKI'

            d = [levenshtein('wanita', str(val['label']).lower()) for val in value_words]
            if(len(d)>0 and min(d) <= 2):
                return 'WANITA'

            d = [levenshtein('perempuan', str(val['label']).lower()) for val in value_words]
            if(len(d)>0 and min(d) <= 2):
                return 'PEREMPUAN'

            return None

    elif(field_name == 'gol_darah'):
            print("ini field golongan darah")
            vals = [val['label'] for val in value_words if len(val['label']) <= 3]
            print(f"vals goldar : {vals}")
            if(len(vals)>0):
                return vals[0]
            else:
                return None

    elif(field_name == 'pekerjaan'):
            d = [levenshtein('kartu', str(val['label']).lower()) for val in value_words]
            if(len(d)>0 and min(d) <= 2):
                idx = np.argmin(d)
                value_words.pop(idx)

            value_words = [val for val in value_words if val['x1'] <= max_x]

    elif(field_name == 'agama'):
            d = [levenshtein('kartu', str(val['label']).lower()) for val in value_words]
            if(len(d)>0 and min(d) <= 2):
                idx = np.argmin(d)
                value_words.pop(idx)

            value_words = [val for val in value_words if val['x1'] <= max_x]


    elif(field_name == 'kewarganegaraan'):
            d = [levenshtein('wni', str(val['label']).lower()) for val in value_words]
            if(len(d)>0):
                return 'WNI'

            xlocs = [val['x1'] for val in value_words]
            if(len(xlocs)>0):
                idx = np.argmin(xlocs)
                return value_words[idx]['label']
            else:
                return None


    elif(field_name == 'status_perkawinan'):
            xlocs = [val['x1'] for val in value_words]
            print(f'index kawin : {index}')
            if(len(xlocs)>0):
                idx = np.argmin(xlocs)
                print(f'idx kawin : {idx}')
                field_value = value_words[idx]['label']
                print(f'field_value kawin : {field_value}')

                if(levenshtein('belum',field_value.lower()) <= 1):
                    return 'BELUM KAWIN'
                elif(levenshtein('belum kawin',field_value.lower()) <= 1):
                    return 'BELUM KAWIN'
                elif(levenshtein('kawin',field_value.lower()) <= 1):
                    return 'KAWIN'
                else:
                    return field_value
            else:
                return None


    elif(field_name == 'berlaku_hingga'):
            d = [levenshtein('hingga', str(val['label']).lower()) for val in value_words]
            if(len(d)>0 and min(d) <= 2):
                idx = np.argmin(d)
                value_words.pop(idx)

            xlocs = [val['x1'] for val in value_words]
            if(len(xlocs)>0):
                idx = np.argmin(xlocs)
                field_value = value_words[idx]['label']
                if(levenshtein('seumur',field_value.lower()) <= 2):
                    return 'SEUMUR HIDUP'
                else:
                    return field_value
            else:
                return None

    field_value = ""    
    for val in value_words:
        field_value = field_value + ' '+ str(val['label'])
    field_value = field_value.lstrip()
    print(f'field_value strip : {field_value}')

    # tes
    field_value_new = value_words_new + " " + field_value 
    print(f'field_value new no strip : {field_value_new}')
    field_value_new_strip = value_words_new + " " + field_value.lstrip() 
    print(f'field_value new strip : {field_value_new_strip}')
    
    

    if(provinsi_search):
        awalan_provinsi = awalan_provinsi.replace("provinsi","")
        field_value = awalan_provinsi.upper()+" "+field_value

    if(provinsi_search_angin):
        # awalan_provinsi = awalan_provinsi.replace("provinsi","")
        prov = prov.replace("provinsi","")
        print(f'prov : {prov}')
        field_value = prov.upper()+" "+field_value
        print(f'field_value : {field_value}')


    return field_value

def get_gender(ls_word):
    """mendapatkan gender dari hasil ocr"""
    new_ls_word = np.asarray([word['label'].lower() for word in ls_word])

    d = [levenshtein('laki-laki', word.lower()) for word in new_ls_word]
    if(len(d)>0 and min(d) <= 3):
            return 'male'

    d = [levenshtein('wanita', word.lower()) for word in new_ls_word]
    if(len(d)>0 and min(d) <= 2):
            return 'female'

    d = [levenshtein('perempuan', word.lower()) for word in new_ls_word]
    if(len(d)>0 and min(d) <= 2):
            return 'female'

    d = [levenshtein('pria', word.lower()) for word in new_ls_word]
    if(len(d)>0 and min(d) <= 1):
            return 'male'

    d = [levenshtein('laki', word.lower()) for word in new_ls_word]
    if(len(d)>0 and min(d) <= 1):
            return 'male'

    return None

def extract_date(date_string):
    """return tanggal lahir"""
    if(date_string == None):
        return None

    date = None
    try:
        regex = re.compile(r'(\d{1,2}-\d{1,2}-\d{1,4})')
        tgl = re.findall(regex, date_string)
        if(len(tgl)>0):
            date = datetime.strptime(tgl[0], '%d-%m-%Y')
        else:
            tgl = ''.join([n for n in date_string if n.isdigit()])
            if(len(tgl)==8):
                date = datetime.strptime(tgl[0:2]+'-'+tgl[2:4]+'-'+tgl[4:], '%d-%m-%Y')
    except ValueError:
        return None

    if(date==None):
        return None

    if((date.year < 1910) or (date.year > 2100)):
        return None

    return date

def find_occupation(occ):
    """mendapatkan value pekerjaan dari ktp"""
    if(occ==None):
        return None
    result = occ
    if(levenshtein('mengurus rumah tangga',occ.lower()) <= 6):
            result = 'Mengurus Rumah Tangga'
    if(levenshtein('buruh harian lepas',occ.lower()) <= 6):
            result = 'Buruh Harian Lepas'
    if(levenshtein('pegawai negeri sipil',occ.lower()) <= 5):
            result = 'Pegawai Negeri Sipil'
    if(levenshtein('pelajar/mahasiswa',occ.lower()) <= 4):
            result = 'Pelajar/Mahasiswa'
    if(levenshtein('pelajar/mhs',occ.lower()) <= 3):
            result = 'Pelajar/Mahasiswa'
    if(levenshtein('mahasiswa',occ.lower()) <= 3):
            result = 'Mahasiswa'
    if(levenshtein('belum/tidak bekerja',occ.lower()) <= 5):
            result = 'Belum/Tidak Bekerja'
    if(levenshtein('karyawan swasta',occ.lower()) <= 4):
            result = 'Karyawan Swasta'
    if(levenshtein('pegawai negeri',occ.lower()) <= 4):
            result = 'Pegawai Negeri'
    if(levenshtein('wiraswasta',occ[0:10].lower()) <= 3):
            result = 'Wiraswasta'
    if(levenshtein('peg negeri',occ.lower()) <= 3):
            result = 'Pegawai Negeri'
    if(levenshtein('peg swasta',occ.lower()) <= 3):
            result = 'Pegawai Swasta'

    return result
def extract_ktp_data(text_response,debug_mode=False):
    """mendapatkan data dari gambar ktp"""
    ktp_extract = pd.DataFrame(columns=['province','city','identity_number','fullname','birth_place',
                                        'birth_date','nationality','occupation','gender','blood_type',
                                        'marital_status','address','rt_rw','kel_desa','kecamatan','religion','expired_date',
                                        'state'])

    attributes = {}

    ls_word = easyocr_format_to_gcv_format(text_response)
    print(f"ls word : {ls_word} \n")

    if(len(ls_word)==0):
        attributes['state'] = "REJECTED"
        ktp_extract = ktp_extract.append(attributes,ignore_index=True)
        return ktp_extract

    global max_x
    max_x = 9999

    raw_result = {}

    for field in fields_ktp:
        field_value = get_attribute_ktp(ls_word,field['field_name'],field['keywords'],field['typo_tolerance'],False)
        print(f'field_value awal raw results : {field_value}')
        if(field_value != None):
            field_value = str(field_value).replace(': ','').replace(':','')
        else:
            field_value = None

        raw_result[field['field_name']] = field_value
        print(f'field_value last / raw results : {field_value}')
        print('\n')
  
    ## Filter kata kata selain 6 agama (terkadang ada kata yang keambil karena ktp miring sehingga sudut antara bounding box agama dengan selain "intended bounding box" terambil)
    religion_list = ["islam","kristen", "protestan","katolik","hindu","buddha","khonghucu", "christian"]


    attributes['identity_number'] = raw_result['nik']
    if(attributes['identity_number'] != None):
        attributes['identity_number'] = ''.join([i for i in raw_result['nik'] if i.isdigit()])

    if(attributes['identity_number'] == None):
        attributes['state'] = "REJECTED"
        ktp_extract = ktp_extract.append(attributes,ignore_index=True)
        return ktp_extract

    attributes['fullname'] = raw_result['nama']
    if(raw_result['nama'] != None):
        attributes['fullname'] = ''.join([i for i in raw_result['nama'] if not i.isdigit()]).replace('-','').strip()

    if(raw_result['jenis_kelamin'] == 'LAKI-LAKI'):
        attributes['gender'] = 'male'
    elif(raw_result['jenis_kelamin'] in ['WANITA','PEREMPUAN']):
        attributes['gender'] = 'female'
    else:
        attributes['gender'] = get_gender(ls_word)

    attributes['birth_place'] = None
    attributes['birth_date'] = None

    if(raw_result['ttl'] != None):
        ttls = raw_result['ttl'].split(', ')
        if(len(ttls)>=2):
            attributes['birth_place'] = ttls[0]
            attributes['birth_date'] = extract_date(ttls[1])

        elif(len(ttls)==1):
            attributes['birth_place'] = ttls[0]

        if(attributes['birth_date'] == None):
            attributes['birth_date'] = extract_date(raw_result['ttl'])

    if(attributes['birth_place'] != None):
        attributes['birth_place'] = ''.join([i for i in attributes['birth_place'] if not i.isdigit()]).replace('-','').replace('.','').strip()

    
    attributes['nationality'] = raw_result['kewarganegaraan']

    if(attributes['nationality'] == "WNI"):
        attributes['nationality'] = "INDONESIA"

    # alter_kawin = ["kawin", "belum kawin", "tidak kawin", "janda", "duda", "cerai"]
    attributes['marital_status'] = raw_result['status_perkawinan']
    result_kawin = raw_result['status_perkawinan']
    print(f"raw results perkawinan = {result_kawin}")
    
    if(raw_result['status_perkawinan'] != None):
        for alter in alter_kawin:
            if(levenshtein(alter, result_kawin.lower()) <= 5):
                    attributes['marital_status'] = alter
                    if(alter == 'belum kawin' or alter == 'tidak kawin'):
                        attributes['marital_status'] = 'SINGLE'
                    elif(alter == 'kawin'):
                        attributes['marital_status'] = 'MARRIED'
                    elif(alter == 'janda' or alter == 'duda' or alter == 'cerai'):
                        attributes['marital_status'] = 'WIDOWED'
                    else:
                        attributes['marital_status'] = None
                    alter_english = attributes['marital_status']  
                    print(f"alter perkawinan = {alter_english}")
                    break
            else:
                attributes['marital_status'] = None

    if(raw_result['agama'] != None):
        for split in religion_list:
            if(levenshtein(split, raw_result['agama'].lower()) <= 2):
                    attributes['religion'] = split.upper()
                    break
            else:
                attributes['religion'] = None
                continue
    else:
        attributes['religion'] = None
    

    if(raw_result['gol_darah'] != None):
        darah =raw_result['gol_darah']
        print(f"gol darah raw : {darah}")
        attributes['blood_type'] = ''.join([i for i in raw_result['gol_darah']]).strip()
        blood = attributes['blood_type']
        print(f"gol darah attribute : {blood}")
        for alter in alter_gol_darah:
            if not attributes['blood_type'].isdigit():
                if(levenshtein(alter, attributes['blood_type']) < 2):
                    if len(attributes['blood_type'])> 1:
                        if attributes['blood_type'][0] == alter or attributes['blood_type'][1] == alter:
                            print(f"gol darah {alter}")
                            attributes['blood_type'] = alter

            elif attributes['blood_type'].isdigit():
                if(attributes['blood_type'] == '0'):
                    print("gol darah O")
                    print("tes")
                    attributes['blood_type'] = 'O'
                # elif(attributes['blood_type'] == '0+'):
                #     print("gol darah O+")
                #     attributes['blood_type'] = 'O+'
                # elif(attributes['blood_type'] == '4'):
                #     print("gol darah A")
                #     attributes['blood_type'] = 'A'
                # elif(attributes['blood_type'] == '4B'):
                #     print("gol darah AB")
                #     attributes['blood_type'] = 'AB'
        if(attributes['blood_type'] not in alter_gol_darah):
            attributes['blood_type'] = None
    else:
        attributes['blood_type'] = None

    # if (len(attributes['birth_place'].split()) > 1 and attributes['blood_type'] != None):
    #     birth_place = attributes['birth_place'].split()
        
    #     birth_place.remove(attributes['blood_type'])
    #     birth_place = " ".join(birth_place)
    
    #     attributes['birth_place'] = birth_place
    

    attributes['occupation'] = find_occupation(raw_result['pekerjaan'])
    attributes['province'] = raw_result['provinsi']
    attributes['city'] = raw_result['kota']
    attributes['address'] = raw_result['alamat']
    attributes['rt_rw'] = raw_result['rt_rw']
    attributes['kel_desa'] = raw_result['kel_desa']
    attributes['kecamatan'] = raw_result['kecamatan']
    attributes['expired_date'] = raw_result['berlaku_hingga']


    print(f"attributes end : {(attributes)}")

    attributes['state'] = "OK"

# if attributes None is too many state rejected
    NULL_TOLERANCE = 3
    count_attributes = []
    for k,v in attributes.items():
        if attributes[k] is None:
            count_attributes.append(str(attributes[k]))
            print(count_attributes)
            print(len(count_attributes))
            if len(count_attributes) >= NULL_TOLERANCE:
                print("rejected state")
                attributes['state'] = "REJECTED"
                rejected = attributes['state']
                print(f"rejected state : {rejected}")
                break

    ktp_extract = ktp_extract.append(attributes,ignore_index=True)
    print(f"ktp_extract end : {ktp_extract}")
    return ktp_extract


def ktp_to_csv(reader,img_path):
    """mengubah data ktp menjadi dictionary"""
    bounds = reader.readtext(img_path,width_ths=0.05)
    ktp_extract = extract_ktp_data(bounds)
    dict_response = dict()
    print(f"ktp extract data : {ktp_extract.columns.tolist()}")
    for data_ktp in ktp_extract.columns.tolist():
        print(f"data ktp : {data_ktp}")
        print(f"type data ktp : {type(data_ktp)}")
        if data_ktp == "birth_date":    
            print("NoneType not Allowed to see this")
            np_date64 = pd.to_datetime(ktp_extract[data_ktp].values[0])
            print(f"np_date64 : {np_date64}")
            if np_date64 != None:
                dict_response[data_ktp] = np.datetime64(np_date64).astype(datetime).strftime("%d/%m/%Y")
        else:
            dict_response[data_ktp] = ktp_extract[data_ktp].values[0]
    return bounds,dict_response