from abc import ABC, abstractmethod

class AbstractDataInput(ABC):
    """
    Defines the commun structure of the data sources
    """

    @abstractmethod
    def get_data(self, tickers, start_date,  end_date, frequency):
        """
        Generic method to retrieve the data (prices) from the source
        """
        pass

    @abstractmethod
    def get_PER(self, tickers, start_date,  end_date, frequency):
        """
        Generic method to retrieve the PER from the source (ONLY FOR EQUOTIES)
        """
        pass