import json
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class ForexDetails(BaseModel):
    amount: float | None = Field(None, description="Amount to convert")
    from_currency: str | None = Field(None, description="Currency to convert from")
    to_currency: str | None = Field(None, description="Currency to convert to")

class ForexConverter(BaseTool):
    name = "Forex Converter"
    description = "Convert currencies based on current exchange rates"
    currency_data: Dict[str, Dict] = Field(default_factory=dict)
    currency_mapping: Dict[str, List[str]] = Field(default_factory=dict)
    reverse_mapping: Dict[str, str] = Field(default_factory=dict)
    unit_mapping: Dict[str, str] = Field(default_factory=dict)
    

    def __init__(self):
        super().__init__()
        self.currency_data = self.load_currency_data()
        self.currency_mapping = self.load_currency_mapping()
        self.reverse_mapping = self.create_reverse_mapping()
        self.unit_mapping = self.load_unit_mapping()
        


    def load_currency_data(self) -> Dict[str, Dict]:
        with open('src/data/exchange_rates.json', 'r') as f:
            data = json.load(f)
        return {item['Currency']: item for item in data}

    def load_currency_mapping(self) -> Dict[str, List[str]]:
        return {
            "USD": ["US Dollar", "Dollar","dollars","$", "USD"],
            "EUR": ["Euro", "€", "EUR"],
            "GBP": ["British Pound", "Pound Sterling", "£", "GBP"],
            "CHF": ["Swiss Franc", "CHF"],
            "AUD": ["Australian Dollar", "AUD"],
            "CAD": ["Canadian Dollar", "CAD"],
            "SGD": ["Singapore Dollar", "SGD"],
            "JPY": ["Japanese Yen", "Yen", "¥", "JPY"],
            "CNY": ["Chinese Yuan", "Renminbi", "¥", "CNY"],
            "HKD": ["Hongkong Dollar", "HKD"],
            "DKK": ["Danish Kroner", "DKK"],
            "MYR": ["Malaysian Ringgit", "MYR"],
            "QAR": ["Qatari Riyal", "QAR"],
            "SAR": ["Saudi Rial", "SAR"],
            "SEK": ["Swedish Kroner", "SEK"],
            "THB": ["Thai Bhat", "THB"],
            "AED": ["UAE Dirham", "AED"],
            "KWD": ["Kuwaiti Dinar", "KWD"],
            "BHD": ["Bahrain Dinar", "BHD"],
            "KRW": ["Korean Won", "KRW"],
            "INR": ["Indian Rupees", "₹", "INR", "Rupees"],
            "NPR": ["Nepali Rupees", "NPR", "रू", "Nepalese Rupee","nrs"],
        }
    
    def load_unit_mapping(self) -> Dict[str, int]:
        return {
            "JPY": 10,
            "KRW": 100,
        }

    def create_reverse_mapping(self) -> Dict[str, str]:
        reverse_map = {}
        for key, values in self.currency_mapping.items():
            for value in values:
                reverse_map[value.lower()] = key
        return reverse_map

    def get_currency_code(self, currency: str) -> str:
        currency = currency.lower()
        for code, aliases in self.currency_mapping.items():
            if currency in [alias.lower() for alias in aliases] or code.lower() == currency:
                return code
        return currency.upper()


    def get_exchange_rate(self, from_currency: str, to_currency: str) -> float:
        if from_currency == to_currency:
            return 1.0
        
        from_code = self.get_currency_code(from_currency)
        to_code = self.get_currency_code(to_currency)

        from_unit = self.unit_mapping.get(from_code, 1)
        to_unit = self.unit_mapping.get(to_code, 1)

        if from_code == 'NPR':
            return to_unit / self.currency_data[self.currency_mapping[to_code][0]]['Selling/Rs.']
        elif to_code == 'NPR':
            return self.currency_data[self.currency_mapping[from_code][0]]['Selling/Rs.'] / from_unit
        else:
            npr_to_from = self.currency_data[self.currency_mapping[from_code][0]]['Selling/Rs.'] / from_unit
            npr_to_to = self.currency_data[self.currency_mapping[to_code][0]]['Selling/Rs.'] / to_unit
            return npr_to_from / npr_to_to 


    def extract_forex_details(self, query: str) -> ForexDetails:
        amount_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:k|thousand|million|m)?', query, re.IGNORECASE)
        from_currency_match = re.search(r'(\w+)\s+to', query, re.IGNORECASE)
        to_currency_match = re.search(r'to\s+(\w+)', query, re.IGNORECASE)
        
        amount = float(amount_match.group(1)) if amount_match else None
        from_currency = self.get_currency_code(from_currency_match.group(1)) if from_currency_match else None
        to_currency = self.get_currency_code(to_currency_match.group(1)) if to_currency_match else None
        
        return ForexDetails(amount=amount, from_currency=from_currency, to_currency=to_currency)



    def _run(self, query: str) -> str:
        forex_details = self.extract_forex_details(query)

        if forex_details.amount is None or forex_details.from_currency is None or forex_details.to_currency is None:
            missing = []
            if forex_details.amount is None:
                missing.append("amount")
            if forex_details.from_currency is None:
                missing.append("source currency")
            if forex_details.to_currency is None:
                missing.append("target currency")
            return f"I need more information. Please provide the {' and '.join(missing)}."

        amount = forex_details.amount
        from_currency = forex_details.from_currency
        to_currency = forex_details.to_currency

        try:
            rate = self.get_exchange_rate(from_currency, to_currency)
            converted_amount = amount * rate
            
            return f"""
            Based on the provided information:
            Amount: {amount} {from_currency}
            From: {from_currency}
            To: {to_currency}
            Exchange Rate: 1 {from_currency} = {rate:.4f} {to_currency}
            
            Converted amount: {round(converted_amount, 2)} {to_currency}
            """
        except KeyError:
            return f"Sorry, I don't have exchange rate information for {from_currency} or {to_currency}."

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("ForexConverter does not support async")

forex_tool = ForexConverter()

