import math
import numpy as np

class ErrorFuncs:
    @staticmethod
    def nrmse(v: np.array, vhat: np.array):
        '''
        takes the Normalised Root Mean Square Error of two output arrays normalised over the standard deviation of the expected output
        v: real output 
        vhat: desired output
        '''
        N = v.size
        if N==0:
            return 
        # print(N)
        if v.size != vhat.size:
            raise ValueError(f"{v.size =}, {vhat.size=}")
        sq = np.power((vhat-v), 2)
        # print(sq.size)
        sqflat = sq.flatten()
        # print(sqflat.shape)
        sumsq = np.sum(sqflat)
        vhatmean = sum(vhat) / N
        vhatminusvhatmeansq = sum((vhat - np.asarray([vhatmean] * N))**2)
        # print(f"sq: {sq}\nsumsq: {sumsq}\nvhatmean: {vhatmean}\nvhatminusmeansq: {vhatminusvhatmeansq}")
        res = math.sqrt(sumsq / vhatminusvhatmeansq)
        # print(res)
        # print(res)
        return res

    @staticmethod
    def nmse(v, vhat):
        '''
        takes the Normalised Mean Square Error of two output arrays normalised over the standard deviation of the expected output
        v: real output
        vhat: desired output
        '''
        N = len(v)
        sq = (vhat - v)**2
        sumsq = sum(sq)
        vhatmean = sum(vhat) / N
        vhatminusvhatmeansq = sum((vhat - np.asarray([vhatmean] * N))**2)
        # print(f"sq: {sq}\nsumsq: {sumsq}\nvhatmean: {vhatmean}\nvhatminusmeansq: {vhatminusvhatmeansq}")
        res = sumsq / vhatminusvhatmeansq
        return res

    @staticmethod
    def mse(v, vhat):
        '''
        takes the Mean Square Error of two output arrays
        v: real output
        vhat: desired output
        '''
        N = len(v)
        sq = (vhat - v)**2
        sumsq = sum(sq.flatten())
        res = sumsq / N
        return res

    @staticmethod
    def rmse(v, vhat):
        '''
        takes the Root Mean Square Error of two output arrays
        v: real output
        vhat: desired output
        '''
        N = len(v)
        sq = (vhat - v)**2
        sumsq = sum(sq.flatten())
        res = math.sqrt(sumsq / N)
        return res
        

    @staticmethod
    def wer(v, vhat):
        pass
    