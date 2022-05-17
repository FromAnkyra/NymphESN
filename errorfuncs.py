import math
import numpy as np

class ErrorFuncs:
    @staticmethod
    def nmsre(v: np.array, vhat: np.array):
        # print(v) #nan
        # print(vhat) # nan brr
        N = v.size
        sq = np.power((vhat-v), 2)
        sumsq = sum(sq.flatten())
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
    