#please read your_code.py

import csv

class ResultsSave():
    def __init__(self,vision_file_name,plc_file_name):

        vision_file=open(vision_file_name, mode='w')
        fieldnames = ['tray', 'part', 'type', 'is good?', 'description']
        self.writer_vision = csv.DictWriter(vision_file, fieldnames=fieldnames)
        self.writer_vision.writeheader()

        plc_file=open(plc_file_name, mode='w')
        fieldnames = ['tray','message']
        self.writer_plc = csv.DictWriter(plc_file, fieldnames=fieldnames)
        self.writer_plc.writeheader()

    def insert_vision(self,tray,part,ty,isGood,fault):

        item={}
        item['tray']=tray
        item['part']=part
        item['type']=ty
        item['is good?']=isGood
        item['description']=fault

        self.writer_vision.writerow(item)


    def insert_plc(self,tray,message):

        item={}
        item['tray']=tray
        item['message']=message

        self.writer_plc.writerow(item)