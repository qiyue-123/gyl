#

在供应链管理课程中有一个任务， 设计一个基于生成式AI的工具优化某个特定供应链管理业务场景：
1）选定一个具体的供应链管理业务场景，分析其痛点；
2）设计解决方案并利用现有的生成式AI工具或平台开发原型系统 要求在查阅文献、资料的基础上，通过思考、研发完成。

目前选定场景为医药电商需求预测，方向为智能决策/问答，使用ChatGPT API等。
有篇论文见 `o2o2.html` 。

```b
pip install -r requirements.txt
set OPENAI_API_KEY=你的API密钥
set OPENAI_BASE_URL=https://api.openai.com/v1
streamlit run app.py
```
