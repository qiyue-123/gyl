import os
import sys
from typing import Optional

try:
	from openai import OpenAI
	from openai.types.chat import ChatCompletion
except Exception as e:
	print("依赖导入失败，请先安装 requirements.txt 中的依赖。")
	raise


def get_api_key() -> Optional[str]:
	"""从环境变量读取 OpenAI API Key。"""
	return os.environ.get("OPENAI_API_KEY")


def test_chat(prompt: str = "用一句话介绍你自己。") -> str:
	"""
	进行一次最小化的 Chat 调用并返回模型回复。

	如果没有配置 OPENAI_API_KEY，会抛出异常。
	"""
	api_key = get_api_key()
	if not api_key:
		raise RuntimeError(
			"未检测到 OPENAI_API_KEY，请在环境变量中设置后重试。"
		)

	client = OpenAI(base_url="https://api.chatanywhere.tech/v1",api_key=api_key)

	# 选择一个性价比和稳定性较好的模型
	model_name = "gpt-4o-mini"

	try:
		completion: ChatCompletion = client.chat.completions.create(
			model=model_name,
			messages=[
				{"role": "system", "content": "你是一个乐于助人的中文助手。"},
				{"role": "user", "content": prompt},
			],
			temperature=0.7,
		)
	except Exception as e:
		# 打印更友好的错误信息
		raise RuntimeError(f"调用 OpenAI API 失败：{e}")

	content = completion.choices[0].message.content or ""
	return content.strip()


def main(argv: list[str]) -> int:
	prompt = "用一句话介绍你自己。"
	if len(argv) > 1:
		prompt = " ".join(argv[1:])

	try:
		reply = test_chat(prompt)
	except Exception as e:
		print(e)
		return 1

	print("模型回复：")
	print(reply)
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv))

