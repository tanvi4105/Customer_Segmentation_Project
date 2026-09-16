from pydantic import BaseModel

class CustomerInput(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float
    AvgOrderValue: float
    TotalQuantity: float
    Country: float