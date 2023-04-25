# 适用于chatrwkv的简单http fastapi
## 介绍
根据Roleplay webui大改，重新封装了RWKV模型调用，重新设计了人设模块与加载方式
根据cryscan的pr修复了脑补对话的bug，并修复了多轮对话复读的bug，
目前个人用于bot搭建，因此仅包含重置和直接对话的功能

## 启动
在main.py里更改模型地址、策略（详见rwkv）和人设文件(json格式，可根据Base.json修改)

使用
```
uvicorn main:app --post 端口(可选) --host IP(可选)
```
启动
