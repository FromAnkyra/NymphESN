import math
import numpy as np

class ErrorFuncs:
    @staticmethod
    def nrmse(v: np.array, vhat: np.array):
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
        N = len(v)
        sq = (vhat - v)**2
        sumsq = sum(sq.flatten())
        res = sumsq / N
        return res

    @staticmethod
    def rmse(v, vhat):
        N = len(v)
        sq = (vhat - v)**2
        sumsq = sum(sq.flatten())
        res = math.sqrt(sumsq / N)
        return res
        

    @staticmethod
    def wer(v, vhat):
        pass
    