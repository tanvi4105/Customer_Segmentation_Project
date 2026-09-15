# from pydantic import BaseModel

# class CustomerInput(BaseModel):
#     Recency: float
#     Frequency: float
#     Monetary: float
#     Avg_Order_Value: float
#     Total_Items: float
#     Product_Variety: float
#     Engagement: float

# from pydantic import BaseModel
from pydantic import BaseModel

class CustomerInput(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float
    AvgOrderValue: float
    TotalQuantity: float
    Country: float