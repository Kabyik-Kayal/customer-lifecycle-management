import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from zenml import step


@step
def dynamic_importer() -> str:
    """Dynamically imports data for testing out the model."""
    # Here, we simulate importing or generating some data.
    # In a real-world scenario, this could be an API call, database query, or loading from a file.
    data = {
        "CustomerID":[1,1721.8179628629177],
    	"first_purchase":[2,10216210.675643647],
        "last_purchase":[5,8706307.504886169],
        "frequency":[3,232.72308852591908],
    	"total_amount":[20,127.55477528418574],
        "avg_order_value":[12,1.0437267496016434],
        "recency":[2,132.8952194556411],
        "customer_age":[3,118.26858842954378],
        "purchase_frequency":[2,15.539375931736696],
        "CLTV":[100,892.8834269893001]
    }
									
    df = pd.DataFrame(data)

    # Convert the DataFrame to a JSON string
    json_data = df.to_json(orient="split")

    return json_data