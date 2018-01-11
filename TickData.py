

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import re
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

    def calcVPIN(self, bucketsize=200, sigma=10, winlen=50):
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
        pass


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
        pass

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

    def plot(self, savingpath=''):
        '''
        make a plot of the vpin instance and save it if provided saving path
        :return: none
        '''

        if len(self.Date) == 1:
            return
        titlestr = self.Contract + "\n"
        for date in np.unique(self.Date):
            titlestr += " " + dt.datetime.strftime(date,'%Y%m%d')
        titlestr += '\n'
        titlestr += "Bucket Size = %4.f | VPIN Window = %4.f" % (self.BucketSize, self.WindowLength)
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        # 多天VPIN以日期为MajorTickLabel，小时为minorTickLabel(待实现）
        # 一天VPIN以小时为MajorTickLabel，半小时为minorTickLabel（待实现）
        if np.unique(self.Date).size == 1:
            def truncatemin(x):
                return x.replace(minute=0, second=0, microsecond=0)
            xtimelabels = np.unique(list(map(truncatemin, self.Time.tolist())))
            xmajortickpos = []
            xtimelabelStr = []
            for xtime in xtimelabels:
                xmajortickpos.append(np.argmax(self.Time >= xtime))
                xtimelabelStr.append(dt.datetime.strftime(xtime, '%Y%m%d-%H'))
        else:
            def truncatehour(x):
                return x.replace(hour=0, minute=0, second=0, microsecond=0)
            xtimelabels = np.unique(list(map(truncatehour, self.Time.tolist())))
            xmajortickpos = []
            xtimelabelStr = []
            for xtime in xtimelabels:
                xmajortickpos.append(np.argmax(self.Time >= xtime))
                xtimelabelStr.append(dt.datetime.strftime(xtime, '%Y%m%d'))
        ax.set_xticks(xmajortickpos)
        ax.set_xticklabels(xtimelabelStr)
        ax.tick_params(rotation=45)
        # ax.minorticks_on()
        ax.plot(self.Value)
        ax.set_xlabel('Time Label')
        ax.set_ylabel('VPIN Value')
        ax.grid(True)
        ax.set_title(titlestr)
        if savingpath != '':
            fig.savefig(savingpath+re.sub(' ', '_', titlestr))


class QuoteData(object):
    """Quote Data of Specific Security
    Attributes:
        Contract: A string of security name
        Date: A datetime array of trading dates
        Time: A datetime array of quotes start time
        Open: A float array of Open quote prices
        High: A float array of High quote prices
        Low: A float array of Low quote prices
        Close: A float array of Close quotes prices
        Volume: A float array of quote Volume
        Amount: A float array of quote Amount
        BarLength: An int indicates the time length of a bar based on BarType
        BarType: A str indicates Bar type, values in {'min','hour','day'}

    Methods:
        info: print out Quote Data information
        plot: plot Candles Chart

    """

    def __init__(self, contractname='', length=np.nan, type=''):
        self.Contract = contractname
        self.Date = dt.datetime(1900, 1, 1)
        self.Time = dt.datetime(1900, 1, 1)
        self.Open = np.nan
        self.High = np.nan
        self.Low = np.nan
        self.Close = np.nan
        self.Volume = np.nan
        self.Amount = np.nan
        self.BarLength = length
        self.BarType = type

    def info(self):
        print("Contract = " + self.Contract)
        print("Bar Feature = " + str(self.BarLength) + ' ' + self.BarType)
        DateSet = np.unique(self.Date)
        strlist = ''
        for date in DateSet:
            strlist = strlist + '' + dt.datetime.strftime(date, '%Y%m%d') + ', '
        print("Date in { " + strlist[:-1] + " }")
        print("Time Length = %d" % len(self.Time))
        print("Open, High, Low, Close, Volume, Amount Lengths are \n%7.2f|%7.2f|%7.2f|%7.2f|%7.2f|%7.2f|%7.2f"
              % (self.Open, self.High, self.Low, self.Close, self.Volume, self.Amount))

    def plot(self):
        pass


class SpreadTickData(object):
    """ Spread Data in Tick Level

    Attributes:
        ContractA: A string represents name of contract A
        ContractB: A string represents name of contract B
        TickDate: A datetime array of trading date of every tick
        TickTime: A datetime array of time stamp of every tick
        SpreadValue: A float array of spread value of every tick
        ValueTypeA: A string represents the type of value related to contract A
        ValueTypeB: A string represents the type of value related to contract B

    Methods:
        info: print out the TickData information including contract name, dates included and length of ticks
        readfromCSV: read TickData from CSV files (two types : 1 homemade by Bro Zhao & 2 purchased)
                    type 1 requires the specific trading date and the date of night as well as the contract name
                    type 2 requires only the specific trading date as well as the contract name
        readfromMAT: read TickData from MAT files (made by ZhiZhi)
                    requires the specific trading date as well as the contract name
        calcVPIN: calculate VPIN value

    """

    def __init__(self, TickDataA=TickData(), TickDataB=TickData(), ValueTypeA='LastPrice', ValueTypeB='LastPrice', computeType='minus'):
        self.ContractA = TickDataA.Contract
        self.ContractB = TickDataB.Contract
        if not list(np.unique(TickDataA.TickDate)) == list(np.unique(TickDataB.TickDate)):
            raise ValueError('TickDatas NOT share same trading dates')
        TickDateSet = np.unique(TickDataA.TickDate)
        self.TickTime = np.empty(0)
        self.TickDate = np.empty(0)
        for date in TickDateSet:
            ticklist = np.unique(np.concatenate((TickDataA.TickTime[TickDataA.TickDate == date], TickDataB.TickTime[TickDataB.TickDate == date])))
            self.TickTime = np.concatenate((self.TickTime, ticklist))
            self.TickDate = np.concatenate((self.TickDate, np.full(ticklist.shape, date)))
        Aloc = np.where(np.isin(self.TickTime, TickDataA.TickTime))[0]
        Aloc_absence = np.setdiff1d(range(len(self.TickTime)), Aloc)
        ValueA = np.zeros(self.TickTime.shape)
        ValueA[Aloc] = TickDataA.__getattribute__(ValueTypeA)
        if 0 in Aloc_absence:
            ValueA[Aloc_absence[1:]] = ValueA[Aloc_absence[1:]-1]
        else:
            ValueA[Aloc_absence] = ValueA[Aloc_absence-1]
        Bloc = np.where(np.isin(self.TickTime, TickDataB.TickTime))[0]
        Bloc_absence = np.setdiff1d(range(len(self.TickTime)), Bloc)
        ValueB = np.zeros(self.TickTime.shape)
        ValueB[Bloc] = TickDataB.__getattribute__(ValueTypeB)
        if 0 in Bloc_absence:
            ValueB[Bloc_absence[1:]] = ValueB[Aloc_absence[1:] - 1]
        else:
            ValueB[Bloc_absence] = ValueB[Bloc_absence - 1]
        self.ValueTypeA = ValueTypeA
        self.ValueTypeB = ValueTypeB
        self.SpreadValue = ValueA - ValueB



































if __name__ == '__main__':

    hedgefiles = [r'D:\TickData\ni1805_20171221.csv', r'D:\TickData\ni1805_20171222.csv',
              r'D:\TickData\ni1805_20171225.csv', r'D:\TickData\ni1805_20171226.csv',
              r'D:\TickData\ni1805_20171227.csv', r'D:\TickData\ni1805_20171228.csv',
              r'D:\TickData\ni1805_20171229.csv', r'D:\TickData\ni1805_20180102.csv',
              r'D:\TickData\ni1805_20180103.csv', r'D:\TickData\ni1805_20180104.csv',
              r'D:\TickData\ni1805_20180105.csv', r'D:\TickData\ni1805_20180108.csv',
              r'D:\TickData\ni1805_20180109.csv', r'D:\TickData\ni1805_20180110.csv']

    dutyfiles = [r'D:\TickData\ni1807_20171221.csv', r'D:\TickData\ni1807_20171222.csv',
              r'D:\TickData\ni1807_20171225.csv', r'D:\TickData\ni1807_20171226.csv',
              r'D:\TickData\ni1807_20171227.csv', r'D:\TickData\ni1807_20171228.csv',
              r'D:\TickData\ni1807_20171229.csv', r'D:\TickData\ni1807_20180102.csv',
              r'D:\TickData\ni1807_20180103.csv', r'D:\TickData\ni1807_20180104.csv',
              r'D:\TickData\ni1807_20180105.csv', r'D:\TickData\ni1807_20180108.csv',
              r'D:\TickData\ni1807_20180109.csv', r'D:\TickData\ni1807_20180110.csv']

    DateStrs = ['20171221', '20171222', '20171225', '20171226', '20171227', '20171228',
                '20171229', '20180102', '20180103', '20180104', '20180105', '20180108',
                '20180109', '20180110']
    DatePres = ['20171220', '20171221', '20171222', '20171225', '20171226', '20171227',
                '20171228', '20171229', '20180102', '20180103', '20180104', '20180105',
                '20180108', '20180109']

    hedgedata = []
    dutydata = []
    for i in range(len(DateStrs)):
        htmp = TickData()
        dtmp = TickData()
        htmp.readfromCSV(hedgefiles[i], 1, 'ni1805', DateStrs[i], DatePres[i])
        dtmp.readfromCSV(dutyfiles[i], 1, 'ni1807', DateStrs[i], DatePres[i])
        hedgedata.append(htmp)
        dutydata.append(dtmp)

    # matfiles = [r'D:\Job\WorkinPython\MarketMaking\MatTickData\ni1805_20171208.mat',
    #             r'D:\Job\WorkinPython\MarketMaking\MatTickData\ni1805_20171211.mat',
    #             r'D:\Job\WorkinPython\MarketMaking\MatTickData\ni1805_20171212.mat',
    #             r'D:\Job\WorkinPython\MarketMaking\MatTickData\ni1805_20171213.mat',
    #             r'D:\Job\WorkinPython\MarketMaking\MatTickData\ni1805_20171214.mat']

    matfiles = ['/Users/zhizhilulu/Documents/MarketMaking/MatTickData/ni1805_20171208.mat',
                '/Users/zhizhilulu/Documents/MarketMaking/MatTickData/ni1805_20171211.mat',
                '/Users/zhizhilulu/Documents/MarketMaking/MatTickData/ni1805_20171212.mat',
                '/Users/zhizhilulu/Documents/MarketMaking/MatTickData/ni1805_20171213.mat',
                '/Users/zhizhilulu/Documents/MarketMaking/MatTickData/ni1805_20171214.mat']

    DateStrs = ['20171208', '20171211', '20171212', '20171213', '20171214']
    matL = []
    for i in range(5):
        mattest = TickData()
        mattest.readfromMAT(matfiles[i], 'ni1805', DateStrs[i])
        matL.append(mattest)
