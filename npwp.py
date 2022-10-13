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
    {'field_name': 'npwp', 'keywords': 'npwp', 'typo_tolerance': 2},
    {'field_name': 'nama', 'keywords': 'nama', 'typo_tolerance': 2},
    {'field_name': 'nik', 'keywords': 'nik', 'typo_tolerance': 1},
    {'field_name': 'alamat', 'keywords': 'JL', 'typo_tolerance': 1},
    {'field_name': 'kel/kec', 'keywords': 'JL', 'typo_tolerance': 2},
    {'field_name': 'prov/kota', 'keywords': 'kabupaten', 'typo_tolerance': 2},
]

alter_alamat = ["jl.", "Jalan","komplek","dusun"]

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

def get_attribute_ktp(ls_word,field_name,field_keywords,typo_tolerance, debug_mode=False):
    """mendapatkan attribut dari foto ktp"""
    
    print(f'field name  : {field_name}')
    print(f'field keywords  : {field_keywords}')

    for index, word in enumerate(ls_word):
        global field_value
        print(f'word : {word}')
        print(f"word : {word['label']}")
        a = word['label'].split(' ')
        print(f"a : {a}")
        for field in fields_ktp:
            z = field_keywords
            print(f"field_keywords : {z}")
            
            ls_dist = [levenshtein(field_keywords, word.lower()) for word in a]

            print(f'ls_dist : {ls_dist}')
            print(f'np.min ls_dist : {np.min(ls_dist)}')
            word_sentence = [word.lower() for word in a]
            print(f'word_sentence : {word_sentence}')

            if np.min(ls_dist) < typo_tolerance and field_name == "nik":
                value_words_new = [x.strip() for x in a[-1].split(':')]
                print(f'value_words_new nik : {value_words_new}')
                field_value = value_words_new[-1]
                print(f'field_value nik : {field_value}')

            if np.min(ls_dist) < typo_tolerance and field_name == "npwp":
                value_words_new = [x.strip() for x in a[-1].split(':')]
                print(f'value_words_new npwp : {value_words_new}')
                field_value = value_words_new[-1]
                print(f'field_value npwp : {field_value}')

            
            if np.min(ls_dist) > 1 and field_name == "alamat":
                global index_alamat
                for alter in alter_alamat:
                    ls_dist = [levenshtein(alter, word.lower()) for word in a]
                    if np.min(ls_dist) <= 1:
                        value_words_new = word['label']
                        print(f'value_words_new alamat: {value_words_new}')
                        field_value = word['label']
                        print(f'field_value alamat : {field_value}')
                        index_alamat = index  
                        print(f'index_alamat  : {index_alamat}')
                        break
                    else:
                        continue
                                
            if(field_name == 'kel/kec'):
                print(f"index alamat kel kec : {index_alamat}")
                value_words_new = ls_word[int(index_alamat)+1]["label"]
                field_value = value_words_new
                print(f'field_value kel/kec: {field_value}')
                
            if(field_name == 'prov/kota'):
                value_words_new = ls_word[index_alamat+2]["label"]
                field_value = value_words_new
                print(f'field_value prov/kota: {field_value}')

            if(field_name == 'nama'):
                value_words_new = ls_word[3]["label"]
                parse_name = [x.strip() for x in value_words_new.split(':')]
                print(f'parse_name : {parse_name}')
                if parse_name[0].lower() == "nama":
                    value_words_new = value_words_new.replace("NAMA","")
                    value_words_new = value_words_new.replace(":","")
                    field_value = value_words_new
                else:
                    field_value = value_words_new
                print(f'field_value : {field_value}')

    return field_value


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
    
def extract_ktp_data(text_response,debug_mode=False):
    """mendapatkan data dari gambar ktp"""
    ktp_extract = pd.DataFrame(columns=['identity_number','npwp_number','fullname/companyname','address','kel/kec','prov/kota','state'])

    attributes = {}

    ls_word = easyocr_format_to_gcv_format(text_response)
    print(f"ls word : {ls_word}")

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
        raw_result[field['field_name']] = field_value
        print(f'field_value last / raw results : {field_value}')
        print('\n')
  
    attributes['fullname/companyname'] = raw_result['nama']

    attributes['npwp_number'] = raw_result['npwp']
    if(attributes['npwp_number'] != None):
        no_npwp = ''.join([i for i in raw_result['npwp'] if i.isdigit()])
    if(attributes['npwp_number'] == None):
        attributes['state'] = "REJECTED"
        ktp_extract = ktp_extract.append(attributes,ignore_index=True)
        return ktp_extract
    if len(no_npwp) <  15:
        attributes['state'] = "REJECTED"
        ktp_extract = ktp_extract.append
    if not no_npwp.isdigit():
        attributes['state'] = "REJECTED"
        ktp_extract = ktp_extract.append(attributes,ignore_index=True)
        return ktp_extract
        
    attributes['identity_number'] = raw_result['nik']
    if(attributes['identity_number'] != None):
        attributes['identity_number'] = ''.join([i for i in raw_result['nik'] if i.isdigit()])

    attributes['address'] = raw_result['alamat']
    attributes['kel/kec'] = raw_result['kel/kec']
    attributes['prov/kota'] = raw_result['prov/kota']


    print(f"attributes end : {(attributes)}")
    print(f"attributes harusnya")

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
    bounds = reader.readtext(img_path,width_ths=10)
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