#!/usr/bin/env python3
#coding=utf-8
import socket
import time
import json
#创建一个socket的类,socket.AF_INET指的是IPV4, SOCK_STREAM指的是TCP协议,UDP协议对应socket_DRAM
# client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

#连接PC服务器的端口,参数是（IP/域名,端口) # 在不同的网络环境下，此ip会不一样
ServerPort=('192.168.3.22',12525)


def send_status_recv_parameter(states ,actions, rewards, next_states, masks):
    #创建一个socket的类,socket.AF_INET指的是IPV4, SOCK_STREAM指的是TCP协议,UDP协议对应socket_DRAM
    client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    data = {
        'states':states,
        'actions':actions,
        'rewards': rewards,
        'next_states': next_states,
        'masks': masks,
    }
    data = json.dumps(data)
    with open('parameters.txt','w+') as f:
        f.write(data)
    # print('data_dumps: ', data)
    res = ''
    # connect_ex是connect()函数的扩展版本,出错时返回出错码,而不是抛出异常
    log = client.connect_ex(ServerPort)
    print('clinet log: ',log)
    while True:
        try:
        # 发送消息，注意用二进制的数据传输,因为发送的实际是一个数据流
            data=((data.encode('utf-8')))
            client.sendall(data)
        # 接受消息，并打印，解码用 utf-8 神经网络的参数
            print("recev model parameter: ")
            res = client.recv(1024).decode('utf-8')
            print(client.recv(1024).decode('utf-8'))
            time.sleep(1)
            client.close()
        except:
            break
    return res


# #接受服务器的消息,限制接受大小为1kb=1024b=128KB
# while True:
#     try:
#     # 发送消息，注意用二进制的数据传输,因为发送的实际是一个数据流
#         data=(('status11234'.encode('utf-8')))
#         client.send(data)

#     #接受消息，并打印，解码用utf-8 神经网络的参数
#         print("accept msg")
#         print(client.recv(1024).decode('utf-8'))
#         time.sleep(2)
#     except:
#         break
# print("Connection is closed by Server")
