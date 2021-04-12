'''
Author: Ful Chou
Date: 2021-03-22 16:14:39
LastEditors: Ful Chou
LastEditTime: 2021-04-02 16:50:52
FilePath: /acIOT/http_client.py
Description: What this document does
'''
from matplotlib.pyplot import step
import requests
import json


def get_parameters():
    req = requests.get('http://192.168.199.128:5000/getParameters')
    return req.text


def multi_list2dict(multi_list):  # 多维list 2 dict
    steps = {}
    for i, step in enumerate(multi_list):
        envs = {}
        for j, env in enumerate(step):
            envs['env{}'.format(j)] = env
        steps['step{}'.format(i)] = envs
    return steps
    # why? 列表推导的问题：
    return {'step{}'.format(i): {'env{}'.format(j): state
                                 for j in range(len(multi_list[i]))
                                 for state in multi_list[i]
                                 }
            for i in range(len(multi_list))
            },


def send_status_recv_parameter(states, actions, rewards, masks, dist_probs):
    status = {'states': multi_list2dict(states),  # 返回了一个 list，取第一个即可
              'actions': multi_list2dict(actions),
              'rewards': multi_list2dict(rewards),
              #   'next_states': multi_list2dict(next_states),
              'masks': multi_list2dict(masks),
              #   'log_probs': multi_list2dict(log_probs),
              #   'entropy':entropy,
              'dist_probs': multi_list2dict(dist_probs),
              }
    status = json.dumps(status)
    req = requests.post('http://192.168.199.128:5000/sendStatus', data=status)
    return json.loads(req.text)


if __name__ == '__main__':
    pass
