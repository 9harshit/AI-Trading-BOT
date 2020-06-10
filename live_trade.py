
import pandas as pd
import requests,json
import statistics as st

    
base_url = "https://paper-api.alpaca.markets"
acnt_url = "{}/v2/account".format(base_url)
orders_url = "{}/v2/orders".format(base_url)

api_key = "PKK34NDWUMPLZYM47HJI"
secret_key = "YDOEGIWSx0nw2ZPiz5/LwGGmnV73K3NIJcPjucmP"
Headers = {'APCA-API-KEY-ID' : api_key, 'APCA-API-SECRET-KEY' :secret_key }


r = requests.get(acnt_url, headers = Headers)
print(json.loads(r.content))



def create_order(symbol, qty, side, typ, time_in_force):
    data = {  "symbol": symbol,
  "qty": qty,
  "type": typ,
  "side": side,
  "time_in_force": time_in_force,
  }
    
    r = requests.post(orders_url, json = data, headers = Headers)

    return json.loads(r.content)



def trade(): 
    
        
    data5 = pd.read_csv("prediction_5min.csv")
    data1 = pd.read_csv("prediction_1min.csv")
    
    
    
    

        if :
            respone = create_order("AAPL", 1, "buy", "market", "day")
            print (respone )
            
        if :
            respone = create_order("AAPL", 1, "sell", "market", "day")
            print (respone )
        
while True:
    sleep(894)
    trade()   