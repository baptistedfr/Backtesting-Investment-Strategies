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
<<<<<<< HEAD
=======
        pass

    def get_PER(self, tickers, start_date,  end_date, frequency):
        """
        Generic method to retrieve the PER from the source (ONLY FOR EQUOTIES)
        """
>>>>>>> b554d17c1efe3b485c957be4060a228f60758895
        pass