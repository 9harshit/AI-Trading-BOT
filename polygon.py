import websocket, json

def on_open(ws):
     print("Connection Opened")
     print("Authenicating...")
     auth_data = {"action":"auth","params":"PKX64L4ELMLHXG1FWS4Q"}
     ws.send(json.dumps(auth_data))
     

     channel_data = {"action":"subcribe","params": "AM.AAPL"}
     ws.send(json.dumps(channel_data))
     
def on_message(ws, msg):
    print("Message Received")
    print(json.loads(msg))
    
    
socket = "wss://alpaca.socket.polygon.io/stocks"
ws = websocket.WebSocketApp(socket,on_open = on_open, on_message = on_message) 

ws.run_forever()

{"action":"auth","params":"PKCYFOUUZI519LVDN99Z"}

import requests
response = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&outputsize=full&interval=1min&apikey=H3JHR05VIDLIPGU0&datatype=csv")
# Print the status code of the response.
print(response.iter_lines()	)


     