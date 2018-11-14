import os
import copy
import pandas as pd
from ..utils.folders import make_folder

FAR = 640
SKY = 95

DEFAULT_FRAME_INFO = {
    'id':None,
    'cactus1':FAR,
    'cactus2':FAR,
    'cactus3':FAR,
    'ptera1x':FAR,
    'ptera1y':SKY,
    'ptera2x':FAR,
    'ptera2y':SKY,
    'ptera3x':FAR,
    'ptera3y':SKY,
    'isJumping':0,
    'isDucking':0
}

DEFAULT_REG = {
    'id':[],
    'cactus1':[],
    'cactus2':[],
    'cactus3':[],
    'ptera1x':[],
    'ptera1y':[],
    'ptera2x':[],
    'ptera2y':[],
    'ptera3x':[],
    'ptera3y':[],
    'isJumping':[],
    'isDucking':[]
}

class GameReg():

    def __init__(self, path, filename):
        self.path = path
        self.filename = filename
        self.reg = copy.deepcopy(DEFAULT_REG)
        self.keys = list(self.reg.keys())
    
    def add(self, frameInfo):
        for k in self.keys:
            self.reg[k].append(frameInfo[k])

    def save(self):
        df = pd.DataFrame(self.reg)
        if len(df) > 600:
            make_folder(self.path)
            df[:-500].to_csv(os.path.join(self.path, self.filename), index=False)