from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import re
import logging

logger = logging.getLogger(__name__)

class LoanDetails(BaseModel):
    amount: float | None = Field(None, description="Loan amount")
    rate: float | None = Field(None, description="Annual interest rate")
    tenure: float | None = Field(None, description="Loan tenure in years")

def extract_loan_details(query: str) -> LoanDetails:
    amount_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:k|thousand|lac|lakh|million|m)?\s*(?:rs\.?|rupees)?', query, re.IGNORECASE)
    rate_match = re.search(r'(\d+(?:\.\d+)?)\s*%', query)
    tenure_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:year|yr|y)', query, re.IGNORECASE)
    
    amount = float(amount_match.group(1)) if amount_match else None
    rate = float(rate_match.group(1)) if rate_match else None
    tenure = float(tenure_match.group(1)) if tenure_match else None
    
    return LoanDetails(amount=amount, rate=rate, tenure=tenure)

class EMICalculator(BaseTool):
    name = "EMI Calculator"
    description = "Calculate EMI (Equated Monthly Installment) for a loan based on natural language input"

    def _run(self, query: str) -> str:
        loan_details = extract_loan_details(query)

        if loan_details.amount is None or loan_details.rate is None or loan_details.tenure is None:
            missing = []
            if loan_details.amount is None:
                missing.append("loan amount")
            if loan_details.rate is None:
                missing.append("interest rate")
            if loan_details.tenure is None:
                missing.append("loan tenure")
            return f"I need more information. Please provide the {' and '.join(missing)}."

        principal = loan_details.amount
        annual_rate = loan_details.rate
        time = loan_details.tenure

        monthly_rate = annual_rate / (12 * 100)
        time_in_months = time * 12
        emi = (principal * monthly_rate * (1 + monthly_rate)**time_in_months) / ((1 + monthly_rate)**time_in_months - 1)
        total_payment = emi * time_in_months
        total_interest = total_payment - principal
        
        return f"""
        Based on the provided information:
        Loan Amount: {principal}
        Annual Interest Rate: {annual_rate}%
        Loan Tenure: {time} years
        
        Your monthly EMI will be: {round(emi, 2)}
        Total amount you will pay over {time} years: {round(total_payment, 2)}
        Total interest you will pay: {round(total_interest, 2)}
        """

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("EMICalculator does not support async")

emi_tool = EMICalculator()
