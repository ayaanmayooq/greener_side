from abc import ABC, abstractmethod
from datetime import date
import pandas as pd

class EODEquitiesProvider(ABC):
    @abstractmethod
    def fetch_history(self, symbol: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        """
        Return daily EOD data with Date index (naive), columns at least:
        open, high, low, close, volume, adj_close (adjusted for splits/divs).
        """
        ...