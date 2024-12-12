import blpapi
import pandas as pd
import numpy as np
from datetime import datetime
from src.tools import FrequencyType
from src.exeptions import FrequencyError, DataError

class BLPApi():
    #-----------------------------------------------------------------------------------------------------    
   
    DATE = blpapi.Name("date")
    ERROR_INFO = blpapi.Name("errorInfo")
    EVENT_TIME = blpapi.Name("EVENT_TIME")
    FIELD_DATA = blpapi.Name("fieldData")
    FIELD_EXCEPTIONS = blpapi.Name("fieldExceptions")
    FIELD_ID = blpapi.Name("fieldId")
    SECURITY = blpapi.Name("security")
    SECURITY_DATA = blpapi.Name("securityData")

    def __init__(self):
        """
            Improve this
            BLP object initialization
            Synchronus event handling
           
        """
        # Create Session object
        self.session = blpapi.Session()
       
       
        # Exit if can't start the Session
        if not self.session.start():
            print("Failed to start session.")
            return
       
        # Open & Get RefData Service or exit if impossible
        if not self.session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return
       
        self.session.openService('//BLP/refdata')
        self.refDataSvc = self.session.getService('//BLP/refdata')
 
        print('Session open')
   
    #-----------------------------------------------------------------------------------------------------
   
    def bdh(self, strSecurity, strFields, startdate, enddate, per='DAILY', perAdj = 'CALENDAR', days = 'NON_TRADING_WEEKDAYS', fill = 'PREVIOUS_VALUE', currency = ""):
        """
            Summary:
                HistoricalDataRequest ;
       
                Gets historical data for a set of securities and fields
 
            Inputs:
                strSecurity: list of str : list of tickers
                strFields: list of str : list of fields, must be static fields (e.g. px_last instead of last_price)
                startdate: date
                enddate
                per: periodicitySelection; daily, monthly, quarterly, semiannually or annually
                perAdj: periodicityAdjustment: ACTUAL, CALENDAR, FISCAL
                curr: string, else default currency is used
                Days: nonTradingDayFillOption : NON_TRADING_WEEKDAYS*, ALL_CALENDAR_DAYS or ACTIVE_DAYS_ONLY
                fill: nonTradingDayFillMethod :  PREVIOUS_VALUE, NIL_VALUE
               
                Options can be selected these are outlined in “Reference Services and Schemas Guide.”    
           
            Output:
                A list containing as many dataframes as requested fields
            # Partial response : 6
            # Response : 5
           
        """
           
        #-----------------------------------------------------------------------
        # Create request
        #-----------------------------------------------------------------------
       
        # Create request
        request = self.refDataSvc.createRequest('HistoricalDataRequest')
       
        # Put field and securities in list is single value is passed
        if type(strFields) == str:
            strFields = [strFields]
           
        if type(strSecurity) == str:
            strSecurity = [strSecurity]
   
        # Append list of securities
        for strF in strFields:
            request.append('fields', strF)
   
        for strS in strSecurity:
            request.append('securities', strS)
   
        # Set other parameters
        request.set('startDate', startdate.strftime('%Y%m%d'))
        request.set('endDate', enddate.strftime('%Y%m%d'))
        request.set('periodicitySelection', per)
        request.set('periodicityAdjustment', perAdj)
        request.set('nonTradingDayFillMethod', fill)
        request.set('nonTradingDayFillOption', days)
        if(currency!=""):  
            request.set('currency', currency)
 
        #-----------------------------------------------------------------------
        # Send request
        #-----------------------------------------------------------------------
 
        requestID = self.session.sendRequest(request)
        print("Sending request")
       
        #-----------------------------------------------------------------------
        # Receive request
        #-----------------------------------------------------------------------
       
        dict_Security_Fields={}
        liste_msg = []
        while True:
            event = self.session.nextEvent()
           
            # Ignores anything that's not partial or final
            if (event.eventType() !=blpapi.event.Event.RESPONSE) & (event.eventType() !=blpapi.event.Event.PARTIAL_RESPONSE):
                continue
           
            # Extract the response message
            msg = blpapi.event.MessageIterator(event).__next__()
            liste_msg.append(msg)
            # Break loop if response is final
            if event.eventType() == blpapi.event.Event.RESPONSE:
                break
   
        #-----------------------------------------------------------------------
        # Exploit data
        #----------------------------------------------------------------------
       
        # Create dictionnary per field
        dict_output = {}
        for field in strFields:
            dict_output[field] = {}
            for ticker in strSecurity:
                dict_output[field][ticker] = {}
                 
        # Loop on all messages
        for msg in liste_msg:
            countElement = 0
            security_data = msg.getElement(self.SECURITY_DATA)
            security = security_data.getElement(self.SECURITY_DATA).getValue() #Ticker
            # Loop on dates
            for field_data in security_data.getElement(self.FIELD_DATA):
               
                # Loop on differents fields
                date = field_data.getElement(0).getValue()
               
                for i in range(1,field_data.numElements()):
                    field = field_data.getElement(i)
                    dict_output[str(field.name())][security][date] = field.getValue()
                   
                countElement = countElement+1 if field_data.numElements()>1 else countElement
                
            # remove ticker
            if countElement==0:
                for field in strFields:
                    del dict_output[field][security]
                   
        for field in dict_output:
            dict_output[field] = pd.DataFrame.from_dict(dict_output[field])
        return dict_output  
    
    def closeSession(self):    
        print("Session closed")
        self.session.stop()

    def _get_freq(self, frequency : str) -> str:
        """
        Map the frequency to Bloomberg accepted frequency str

        Args:
            frequency (str) : frequency selected by the user

        Returns:
            frequency (str) : Bloomberg valid data frequency (DAILY, WEEKLY, MONTHLY)
        """

        match frequency:
            case FrequencyType.MONTHLY:
                return 'MONTHLY'
            case FrequencyType.WEEKLY:
                return 'WEEKLY'
            case FrequencyType.DAILY:
                return 'DAILY'
            case _:
                raise FrequencyError(f"Invalid frequency: {frequency}")
            
    def get_data(self, tickers : list[str], frequency : FrequencyType, start_date : str = '2023-10-01',  end_date : str = '2024-10-01') -> pd.DataFrame :
        freq = self._get_freq(frequency)
        startDate = datetime.strptime(start_date, "%Y-%m-%d")
        endDate = datetime.strptime(end_date, "%Y-%m-%d")
        strFields = ["TOT_RETURN_INDEX_GROSS_DVDS "]
        return self.bdh(strSecurity=tickers, strFields = strFields, startdate = startDate, enddate = endDate, per = freq)