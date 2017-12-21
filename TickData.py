

import numpy as np
import datetime as dt
from scipy.io import loadmat
from scipy.stats import norm
from pandas import read_csv





class TickData(object):
    """Tick Data of Specific Security

       Attributes:
           Contract: A string represent the contract name
           TickDate: A datetime array of trading date of every tick
           TickTime: A datetime array of time stamp of every tick
           LastPrice: A float array of last prices of every tick
           Volume: A float array of volumes of every tick
           Amount: A float array of amounts of every tick
           BidPrice: A float array of bid price of every tick
           BidVolume: A float array of bid volume of every tick
           AskPrice: A float array of ask price of every tick
           AskVolume: A float array of ask volume of every tick
       Methods:
           info: print out the TickData information including contract name, dates included and length of ticks
           readfromCSV: read TickData from CSV files (two types : 1 homemade by Bro Zhao & 2 purchased)
                        type 1 requires the specific trading date and the date of night as well as the contract name
                        type 2 requires only the specific trading date as well as the contract name
           readfromMAT: read TickData from MAT files (made by ZhiZhi)
                        requires the specific trading date as well as the contract name
           calcVPIN: calculate VPIN value

    """

    def __init__(self, Contract=''):
        self.Contract = Contract
        self.TickDate = dt.datetime(1900, 1, 1)
        self.TickTime = dt.datetime(1900, 1, 1)
        self.LastPrice = np.nan
        self.Volume = np.nan
        self.Amount = np.nan
        self.BidPrice = np.nan
        self.AskPrice = np.nan
        self.BidVolume = np.nan
        self.AskVolume = np.nan

    def __getitem__(self, key):
        '''

        :param key: int index or datetime refer to the specific TickData
        :return: numpy array of the specific line of the TickData
        '''
        # get specific line of TickData
        return np.array([self.TickDate[key], self.TickTime[key], self.LastPrice[key],
                         self.Volume[key], self.Amount[key], self.BidPrice[key],
                         self.AskPrice[key], self.BidVolume[key], self.AskVolume[key]])

    def __add__(self, other):
        '''

        :param other: other TickData instances
        :return: TickData instance that is the union of both self and other
        '''
        # concatenate two different DailyTickData instances together
        if not isinstance(other, type(self)):
            raise TypeError('addition components unmatched')
        if self.Contract != other.Contract:
            raise TypeError('addition components have different underlying contract')
        # result initialization
        tmp = TickData()
        tmp.Contract = self.Contract
        # get tickdate not the same in self instance from the other DailyTickData instance
        tmpAset = np.unique(self.TickDate)
        tmpBset = np.unique(other.TickDate)
        tmpRep = tmpBset[np.isin(tmpBset, tmpAset, invert=True)]
        # merge the self and other DailyTickData instances into the result instance
        tmp.TickDate = np.concatenate((self.TickDate, other.TickDate[np.isin(other.TickDate, tmpRep)]), axis=0)
        tmp.TickTime = np.concatenate((self.TickTime, other.TickTime[np.isin(other.TickDate, tmpRep)]), axis=0)
        tmp.LastPrice = np.concatenate((self.LastPrice, other.LastPrice[np.isin(other.TickDate, tmpRep)]), axis=0)
        tmp.Amount = np.concatenate((self.Amount, other.Amount[np.isin(other.TickDate, tmpRep)]), axis=0)
        tmp.Volume = np.concatenate((self.Volume, other.Volume[np.isin(other.TickDate, tmpRep)]), axis=0)
        tmp.BidVolume = np.concatenate((self.BidVolume, other.BidVolume[np.isin(other.TickDate, tmpRep)]), axis=0)
        tmp.AskVolume = np.concatenate((self.AskVolume, other.AskVolume[np.isin(other.TickDate, tmpRep)]), axis=0)
        tmp.BidPrice = np.concatenate((self.BidPrice, other.BidPrice[np.isin(other.TickDate, tmpRep)]), axis=0)
        tmp.AskPrice = np.concatenate((self.AskPrice, other.AskPrice[np.isin(other.TickDate, tmpRep)]), axis=0)
        # resort the result instance regarding to TickTime
        idx = np.argsort(tmp.TickTime)
        tmp.TickDate = tmp.TickDate[idx]
        tmp.LastPrice = tmp.LastPrice[idx]
        tmp.Amount = tmp.Amount[idx]
        tmp.Volume = tmp.Volume[idx]
        tmp.BidPrice = tmp.BidPrice[idx]
        tmp.AskPrice = tmp.AskPrice[idx]
        tmp.BidVolume = tmp.BidVolume[idx]
        tmp.AskVolume = tmp.AskVolume[idx]
        tmp.TickTime = tmp.TickTime[idx]
        # return the result instance
        return tmp

    def info(self):
        ''' print out information and data length of instance of TickData
        '''
        print("Contract = " + self.Contract)
        TickDateSet = np.unique(self.TickDate)
        strlist = ''
        for Date in TickDateSet:
            strlist = strlist + '' + dt.datetime.strftime(Date,'%Y%m%d') +','

        print("TickDate in {" + strlist[:-1] + '}')
        print("TickTime Length = %d\nLastPrice Length = %d" % (self.TickTime.size, len(self.LastPrice)))
        print("Volume Length = %d\nAmount Length = %d\nBidPrice Length = %d" % (len(self.Volume), len(self.Amount), len(self.BidPrice)))
        print("AskPrice Length = %d\nBidVolume Length = %d\nAskVolume Length = %d" % (len(self.AskPrice), len(self.BidVolume), len(self.AskVolume)))

    def readfromCSV(self,csvfile,sourceType,contractname,datestr,DatePre='19000101'):
        '''
        :param csvfile: csvfile name of the tick data
        :param contractname: str of contractname
        :param sourceType: different source : 1 stands for ourself made; 2 stands for purchased
        :param datestr: trading date in format 'yyyymmdd'
        :param DatePre: date of night trading time in format 'yyyymmdd', default = 19000101
        :return: none
        '''

        self.Contract = contractname
        if sourceType == 1:
            # Read csvfile from Bro ZHAO
            csvcontent = read_csv(csvfile)
            self.LastPrice = np.array(csvcontent['LastPrice'])
            self.Volume = np.array(csvcontent['Volume'] - csvcontent['Volume'].shift(1))
            self.Amount = np.array(csvcontent['Turnover'] - csvcontent['Turnover'].shift(1))
            if np.isnan(self.Volume[0]):
                self.Volume[0] = csvcontent['Volume'][0]
                self.Amount[0] = csvcontent['Turnover'][0]
            self.BidPrice = np.array(csvcontent['BidPrice1'])
            self.AskPrice = np.array(csvcontent['AskPrice1'])
            self.BidVolume = np.array(csvcontent['BidVolume1'])
            self.AskVolume = np.array(csvcontent['AskVolume1'])
            def timeprocess(x,DatePre,datestr):
                DatePrenxt = dt.datetime.strftime(dt.datetime.strptime(DatePre, '%Y%m%d') + dt.timedelta(1), '%Y%m%d')
                if int(x[:2]) > 16:
                    return DatePre[:4] + '-' + DatePre[4:6] + '-' + DatePre[6:] + ' ' + x
                elif int(x[:2]) < 3:
                    return DatePrenxt[:4] + '-' + DatePrenxt[4:6] + '-' + DatePrenxt[6:] + ' ' + x
                else:
                    return datestr[:4] + '-' + datestr[4:6] + '-' + datestr[6:] + ' ' + x
            timeline = csvcontent['UpdateTime'].apply(timeprocess, args=(DatePre,datestr)).values
        elif sourceType == 2:
            # Read csvfile from Purchased
            csvcontent = read_csv(csvfile, encoding='gb2312')
            self.LastPrice = np.array(csvcontent['最新'])
            self.Volume = np.array(csvcontent['成交量'])
            self.Amount = np.array(csvcontent['成交额'])
            self.BidPrice = np.array(csvcontent['买一价'])
            self.BidVolume = np.array(csvcontent['买一量'])
            self.AskPrice = np.array(csvcontent['卖一价'])
            self.AskVolume = np.array(csvcontent['卖一量'])
            timeline = csvcontent['时间'].values
        self.TickTime = np.array([dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S.%f') for x in timeline])
        self.TickDate = np.full(self.TickTime.shape, dt.datetime.strptime(datestr, '%Y%m%d'))

    def readfromMAT(self, matfile, contractname, datestr):
        '''
        :param matfile: matfile name of the tick data
        :param contractname: name of the contract
        :param datestr: str of the specific trading date
        :return:
        '''
        self.Contract = contractname
        matcontent = loadmat(matfile)
        def f(x):
            return dt.datetime.strptime(x[0][0], '%Y-%m-%d %H:%M:%S.%f')
        self.TickTime = np.apply_along_axis(f, 1, matcontent['TickTime'])
        self.TickDate = np.full(self.TickTime.shape, dt.datetime.strptime(datestr, "%Y%m%d"))
        self.LastPrice = np.array(matcontent['TickData'][:, 0])
        self.Volume = np.array(matcontent['TickData'][:, 2])
        self.Amount = np.array(matcontent['TickData'][:, 1])
        self.BidPrice = np.array(matcontent['TickData'][:, 3])
        self.AskPrice = np.array(matcontent['TickData'][:, 4])
        self.BidVolume = np.array(matcontent['TickData'][:, 5])
        self.AskVolume = np.array(matcontent['TickData'][:, 6])

    def calcVPIN(self, bucketsize=200, sigma=10, winlen=10):
        # calculate VPIN value on daily bases
        calendar = np.unique(self.TickDate)
        vpintime = np.empty([1, 1])
        vpindate = np.empty([1, 1])
        vpinvalue = np.empty([1, 1])
        for date in calendar:
            # truncate tick data of every data, leave the first and last lines of tick data
            tdDeltaPrice = self.LastPrice[self.TickDate == date][1:-1] - self.LastPrice[self.TickDate == date][:-2]
            tdVolume = self.Volume[self.TickDate == date][1:-1]
            tdTickTime = self.TickTime[self.TickDate == date][1:-1]
            tdcdf = norm.cdf(tdDeltaPrice / sigma)
            tdvpin_time = [self.TickTime[self.TickDate == date][0], ]
            tdvpin_value = np.zeros([1])
            tdvpin_total_vol = np.zeros([1])
            tdvpin_buy_vol = np.zeros([1])
            tdvpin_sell_vol = np.zeros([1])
            for volume, cdf, ticktime in zip(tdVolume, tdcdf, tdTickTime):
                while volume > 0:
                    vacantvol = bucketsize - tdvpin_total_vol[-1]
                    if volume < vacantvol:
                        tdvpin_total_vol[-1] += volume
                        tdvpin_buy_vol[-1] += volume * cdf
                        tdvpin_sell_vol[-1] += volume * (1 - cdf)
                        volume = 0
                    else:
                        tdvpin_time[-1] = ticktime
                        tdvpin_total_vol[-1] += vacantvol
                        tdvpin_buy_vol[-1] += vacantvol * cdf
                        tdvpin_sell_vol[-1] += vacantvol * (1 - cdf)
                        volume -= vacantvol
                        if len(tdvpin_value) >= winlen:
                            tdvpin_value[-1] = np.abs(tdvpin_buy_vol[-winlen:] - tdvpin_sell_vol[-winlen:]).sum() \
                                               / tdvpin_total_vol[-winlen:].sum()
                        tdvpin_time.append(self.TickTime[self.TickDate == date][0])
                        tdvpin_total_vol = np.append(tdvpin_total_vol, 0)
                        tdvpin_buy_vol = np.append(tdvpin_buy_vol, 0)
                        tdvpin_sell_vol = np.append(tdvpin_sell_vol, 0)
                        tdvpin_value = np.append(tdvpin_value, 0)
            tdvpin_time = np.array(tdvpin_time)
            tdvpin_date = np.full(tdvpin_time.shape, date)
            vpintime = np.append(vpintime, tdvpin_time[:-1])
            vpinvalue = np.append(vpinvalue, tdvpin_value[:-1])
            vpindate = np.append(vpindate, tdvpin_date[:-1])
        vpinfinal = VPINData()
        vpinfinal.Contract = self.Contract
        vpinfinal.WindowLength = winlen
        vpinfinal.BucketSize = bucketsize
        vpinfinal.Date = vpindate[1:]
        vpinfinal.Time = vpintime[1:]
        vpinfinal.Value = vpinvalue[1:]
        return vpinfinal

    def toQuote(self, length = 1, type = ''):

class VPINData(object):
    '''VPIN Data

       Attributes:
           Contract: A string represent the contract name
           Date: A datetime array of trading date
           Time: A datetime array of time when the bucket is filled up and a VPIN value is calculated
           Value: A float array of the VPIN Value of the bucket
           BucketSize: A float specifies how many volume to build up one bucket
           WindowLength: An int specifies the how many buckets to calculate the VPIN value in a moving window form
       Methods:


    '''

    def __init__(self, Contract=''):
        self.Contract = Contract
        self.Date = dt.datetime(1900, 1, 1)
        self.Time = dt.datetime(1900, 1, 1, 0, 0, 0, 0)
        self.Value = np.nan
        self.BucketSize = np.nan
        self.WindowLength = np.nan

    def __getitem__(self, key):
        '''

        :param key: int index or datetime refer to the specific VPIN
        :return:
        '''

    def info(self):
        '''
        print out basic information
        :return:
        '''

        print("Contract = " + self.Contract)
        DateSet = np.unique(self.Date)
        strlist = ''
        for date in DateSet:
            strlist = strlist + '' + dt.datetime.strftime(date, '%Y%m%d') + ','
        print("Date in {" + strlist[:-1] + "}")
        print("Bucket Size = %d\nWindowLength = %d" % (self.BucketSize, self.WindowLength))
        print("Time Length = %d" % len(self.Time))
        print("VPIN Value Length = %d" % len(self.Value))

class QuoteData(object):
    """Quote Data of Specific Security
    Attributes:
        Date: A

    """


if __name__ == '__main__':
    csvfiles = [r'D:\Job\WorkinPython\MarketMaking\TickData\ni1805_20171208.csv',
                r'D:\Job\WorkinPython\MarketMaking\TickData\ni1805_20171211.csv',
                r'D:\Job\WorkinPython\MarketMaking\TickData\ni1805_20171212.csv',
                r'D:\Job\WorkinPython\MarketMaking\TickData\ni1805_20171213.csv',
                r'D:\Job\WorkinPython\MarketMaking\TickData\ni1805_20171214.csv']
    DateStrs = ['20171208','20171211','20171212','20171213','20171214']
    DatePres = ['20171207','20171208','20171211','20171212','20171213']
    csvL = []
    for i in range(5):
        csvtest = TickData()
        csvtest.readfromCSV(csvfiles[i],1,'ni1805',DateStrs[i],DatePres[i])
        csvL.append(csvtest)
    aa = csvL[0].calcVPIN()



    matfiles = [r'D:\Job\WorkinPython\MarketMaking\MatTickData\ni1805_20171208.mat',
                r'D:\Job\WorkinPython\MarketMaking\MatTickData\ni1805_20171211.mat',
                r'D:\Job\WorkinPython\MarketMaking\MatTickData\ni1805_20171212.mat',
                r'D:\Job\WorkinPython\MarketMaking\MatTickData\ni1805_20171213.mat',
                r'D:\Job\WorkinPython\MarketMaking\MatTickData\ni1805_20171214.mat']
    DateStrs = ['20171208', '20171211', '20171212', '20171213', '20171214']
    matL = []
    for i in range(5):
        mattest = TickData()
        mattest.readfromMAT(matfiles[i], 'ni1805', DateStrs[i])
        matL.append(mattest)
