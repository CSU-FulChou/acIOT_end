<!--
 * @Author: Ful Chou
 * @Date: 2021-03-22 16:14:39
 * @LastEditors: Ful Chou
 * @LastEditTime: 2021-04-01 16:43:39
 * @FilePath: /acIOT/readme.md
 * @Description: What this document does
-->
# 开发记录：

-  2021.03.09: client 通过连接一次 server 发送环境env，然后获得 网络模型的 参数进行下一次参数的选择。

- 2021.03.22：
  - tcp 传参遇到 粘包现象，不能拿到一个完整的包，通过搜索，可以拆包解决，但是觉得太麻烦了！
  - 下午采用 flask，进行网络传输
  - client 采用 requests 读取接口的数据， 服务器使用flask 来get，post 状态和参数
  - 遇到新问题，http协议 无法传送一个嵌套的list， 所以把 多维list 拆成 dic，然后进行 序列化json，然后发送给服务端，服务器使用 request.data 进行接受， request.form 无法接受str，它接受的估计是对象吧。
  - 通过函数直接将 多维list 转换成 dic，然后json.dumps， post 2 server

- 2021.03.23:
  - 上午将传输过来的 dic 转换成tensor
  - 下午，从三点开始 晚上运行代码，让client 发送status 后， server 进行训练
  - 发现bug， server训练不会更新 actor 参数， 通过分析得到原因
  - 没有使用 torch 计算图，即没有联系，没有梯度信息。
  - 后续方案，采用 actor 的信息，但是修改actor 需要计算值的构成即可.
- 2021.03.24：
  - 上午 通过传递 dist.probs 参数过去，尝试打印看看，然后赋值， 现在看print，内容一样不知道为什么
  - 下午，写敬业度的 结题论文：
  - 睡到2：40，三点才开始，debug，探究为什么不一样，原来只是打印精度的问题，再说了一开始目标只是探究能否赋值，直接查文档快多了！
  - 四点半开始写敬业度的论文：！
- 2021.03.30:
  - 晚上六点多将近七点开始debug：
  - tensor.data 赋值，通过模拟计算图，然后赋值来使用传递参数
  - 发现bug： 之前字典生成会导致数据错误，重新写 多维数组2dic
  - 发现bug，计算 loss时候维度不对，主要原因是 从dic2多维数组的时候，采用 torch.tensor()直接对数组使用，然后某些 [5,2,1]的数据  总是变成了 [5,2]数据， 简单的[1]  变成了 value （这里得好好研究一下！
  - 重新设计 dic2list， 然后取个 list[i] 转换成 FloatTensor, 然后计算成 [10,1]维度，带入advantage 计算loss
  - bug3， 模型的计算图在第二次运行时消失，我总结是因为我的参数重新load了，所以最好保存一下计算图？
- 2021.03.31:happy::
  - 优化代码，优化更新参数，测试模型训练结果 frame_idx:  20000 [167.45, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 196.9, 200.0, 198.8, 200.0, 200.0, 197.1, 172.4, 140.6, 102.55, 111.95, 116.25] time:  94.31909489631653
  - 发现过拟合问题，寻找解决办法？
  - 下午四点半开会，晓裕师兄说，中间效果差，然后过拟合的现象可以 指定达到某一条线，就可以终止训练就可以了
- 2021.04.01 下午时候：
  - 优化代码，减少参数传递，删除不必要注释
  - 部署actor 到 树莓派上运行，训练结果 frame_idx:  5000 [113.75, 135.2, 199.2, 198.3, 200.0]，发现一个小tip 不能够在本机的env训练模型然后直接给 树莓派加载，这样并不会加快训练速度，反而会导致不收敛，初步怀疑是机器的随机因子问题,那肯定是 机器不同，环境的随机初始seed会不一样
- 后面几天都是在弄 大创论文，寻找一个新的 环境解决办法，无奈，没有能够解决，所以下一步的目标是：
  - TODO： 在 将多个算法去实现分离 这个架构
  - 4.12日下午的主要工作就是整理之前的代码， 分析模型超参数， 计算一个节约参数比： 