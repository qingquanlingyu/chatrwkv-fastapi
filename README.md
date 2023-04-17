# 适用于chatrwkv的简单http fastapi
## 介绍
根据猫娘webui修改，并根据cryscan的pr修复了脑补对话的bug
目前个人仅用于bot搭建，因此仅包含重置和直接对话的功能

## 启动
在main.py里更改模型地址、策略（详见chatrwkv）和人设文件(json格式，可根据Base.json修改)

使用
```uvicorn main:app```
启动
