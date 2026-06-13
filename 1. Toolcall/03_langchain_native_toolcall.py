from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

DEFAULT_USER_PROMPT = "帮我查一下 Beijing 的天气和时间"

SYSTEM_PROMPT = """
你是一个助手。
如果用户询问天气，请调用 get_weather 工具，不要自己编造天气。
如果用户询问时间，请调用 get_time 工具，不要自己编造时间。
""".strip()


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    weather = {
        "Beijing": "晴天，25 度",
        "Shanghai": "多云，28 度",
        "Hangzhou": "小雨，22 度",
    }
    return f"{city} 的天气是：{weather.get(city, '未知天气')}"

@tool
def get_time(city: str) -> str: 
    """Get the current time for a city."""
    time = {
        "Beijing": "14:00",
        "Shanghai": "14:00",
        "Hangzhou": "14:00",
    }
    return f"{city} 的当前时间是：{time.get(city, '未知时间')}"

# @tool
# def get_weather(city: str) -> str:
#     # 郑州经纬度（你可以换成别的城市）
#     lat, lon = 34.75, 113.62

#     url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
#     res = requests.get(url).json()

#     weather = res["current_weather"]
#     return f"温度: {weather['temperature']}°C, 风速: {weather['windspeed']}km/h"


def load_llm() -> ChatOpenAI:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return ChatOpenAI(
        model=os.getenv("MODEL", "agnes-2.0-flash"),
        base_url=os.getenv("BASE_URL", "https://apihub.agnes-ai.com/v1"),
        api_key=SecretStr(os.getenv("API_KEY", "sk-FnTaAp4YjsAwaPVhyySOVhhr4S4LiIpdHfw6p5uYhZqYFHKI")),
        temperature=0,
    )


def main() -> None:
    user_prompt = " ".join(sys.argv[1:]).strip() or DEFAULT_USER_PROMPT

    print("=== 03. LangChain 原生 ToolCall ===")
    print("\n用户请求:")
    print(user_prompt)

    tools = [get_weather, get_time]
    llm = load_llm().bind_tools(tools)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)

    print("\n模型文本输出:")
    print(response.content)

    print("\nLangChain 解析出的 tool_calls:")
    print(response.tool_calls)

    # 如果模型决定调用工具
    if response.tool_calls:
        # 1. 首先，把模型的这个决定（AIMessage）加入历史，只能加一次！
        messages.append(response)
        
        # 2. 准备一个工具映射字典，方便根据名字找到对应的函数
        available_tools = {
            "get_weather": get_weather,
            "get_time": get_time,
        }
        
        # 3. 遍历所有的工具调用
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"\n准备执行工具: {tool_name}, 参数: {tool_args}")
            
            # 根据名字找到对应的工具执行，而不是写死
            selected_tool = available_tools.get(tool_name)
            if selected_tool:
                result = selected_tool.invoke(tool_args)
            else:
                result = f"未找到名为 {tool_name} 的工具"
                
            print(f"工具执行结果: {result}")
            
            # 4. 把每个工具的执行结果作为 ToolMessage 追加进去
            messages.append(
                ToolMessage(
                    content=result,
                    name=tool_name,
                    tool_call_id=tool_call["id"], # 必须带上 id，模型才知道哪个结果对应哪个调用
                )
            )
            
    # 5. 最后，把包含所有工具结果的 messages 再次交给模型做总结
    if response.tool_calls:
        final_response = llm.invoke(messages)
        print("\n把工具结果交回模型后的最终回答:")
        print(final_response.content)


if __name__ == "__main__":
    main()
