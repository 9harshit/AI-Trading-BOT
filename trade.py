#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 14:50:09 2020

@author: harshit
"""


import requests, json

base_url = "https://paper-api.alpaca.markets"
acnt_url = "{}/v2/account".format(base_url)
orders_url = "{}/v2/orders".format(base_url)

api_key = "PKKIEAEC99Y2B7TB04FD"
secret_key = "ITyHutdPfoTAdpvRZcupni207Qy9nc0a64XcO6NQ"
Headers = {'APCA-API-KEY-ID' : api_key, 'APCA-API-SECRET-KEY' :secret_key }


def get_account():
        r = requests.get(acnt_url, headers = Headers)
        return json.loads(r.content)
 

def create_order(symbol, qty, side, typ, time_in_force):
    data = {  "symbol": symbol,
  "qty": qty,
  "type": typ,
  "side": side,
  "time_in_force": time_in_force,
  "stop_limit": 320
  }
    r = requests.post(orders_url, json = data, headers = Headers)

    return json.loads(r.content)

acnt = get_account()
print (acnt)

respone = create_order("AAPL", 1, "buy", "market", "day")
print (respone )