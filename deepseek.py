import os
import re
import json
import readline
from datetime import datetime
from openai import OpenAI
from ftfy import fix_text

class TextSanitizer:
    def __init__(self):
        self.error_log = []

    def sanitize(self, text):
        """多阶段Unicode清理策略"""
        if not text:
            return text

        original = text
        try:
            # 阶段1：修复常见Unicode问题
            text = fix_text(text, normalization='NFKC')
            
            # 阶段2：替换代理对字符
            text = re.sub(
                r'[\ud800-\udfff]',
                lambda m: f'\\u{m.group(0).encode("unicode_escape").decode()[-4:]}',
                text
            )
            
            # 阶段3：严格编码验证
            text = text.encode('utf-8', 'replace').decode('utf-8')
            
            if original != text:
                self._log_error(original, text)
                
            return text
        except Exception as e:
            self._log_error(original, str(e))
            return "[内容包含无法解析的字符]"

    def sanitize_message_history(self, messages):
        """清理整个对话历史"""
        return [{
            "role": msg["role"],
            "content": self.sanitize(msg["content"])
        } for msg in messages]

    def _log_error(self, original, cleaned):
        """记录字符清理日志"""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "original": original.encode('unicode_escape').decode(),
            "cleaned": str(cleaned).encode('unicode_escape').decode()
        }
        self.error_log.append(error_info)
        with open('char_errors.log', 'a', encoding='utf-8') as f:
            json.dump(error_info, f, ensure_ascii=False)
            f.write('\n')

class DeepSeekChat:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.home = os.getenv("DEEPSEEK_HOME")

        if not self.api_key:
            raise ValueError("set environment DEEPSEEK_API_KEY")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self.sanitizer = TextSanitizer()
        self.messages = []
        self._initialize_session()

    def _initialize_session(self):
        """初始化对话会话"""
        self.messages = self.sanitizer.sanitize_message_history([{
            "role": "system",
            "content": "你是一个命令行 AI 助手，回答需要简洁明了，使用中文，并且避免使用 Markdown 格式。请直接回答用户的问题，不要添加额外的说明或格式化内容。"
        }])

    def add_message(self, role, content):
        """安全添加消息到历史"""
        clean_content = self.sanitizer.sanitize(content)
        self.messages.append({
            "role": role,
            "content": clean_content
        })
        # 保持最近10轮对话
        if len(self.messages) > 11:  # 1系统消息 + 10轮对话
            self.messages = [self.messages[0]] + self.messages[-10:]

    def stream_chat(self, max_retries=3):
        """执行带重试机制的流式对话"""
        attempt = 0
        while attempt < max_retries:
            try:
                safe_messages = self.sanitizer.sanitize_message_history(self.messages)
                
                stream = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=safe_messages,
                    stream=True,
                    temperature=0.7,
                    max_tokens=1024,
                    timeout=30
                )
                
                full_response = []
                print("assistant：", end="", flush=True)
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        clean_content = self.sanitizer.sanitize(content)
                        print(clean_content, end="", flush=True)
                        full_response.append(clean_content)
                
                print()  # 确保换行
                self.add_message("assistant", "".join(full_response))
                return
                
            except UnicodeEncodeError as e:
                attempt += 1
                print(f"\n编码错误，正在重试({attempt}/{max_retries})...")
                self.messages = self.sanitizer.sanitize_message_history(self.messages)
            except Exception as e:
                error_msg = self.sanitizer.sanitize(f"\n请求出错：{str(e)}")
                print(error_msg)
                return
        
        print("\n超过最大重试次数，建议开启新对话")

    def should_save_conversation(self):
        """判断是否满足保存条件（用户+助手消息 ≥ 2条）"""
        # messages[0] 是系统消息，后续每轮对话包含用户和助手各1条
        #print(len(self.messages))
        return len(self.messages) >= 5  # 系统消息 + 用户 + 助手 = 3条


    def save_conversation(self):
        """带时间戳的自动保存"""
        if not self.should_save_conversation():
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self._generate_title()}_{timestamp}"

        # 定义目录路径
        directory = f"{self.home}/json"
        # 检测目录是否存在
        if not os.path.exists(directory):
            # 如果不存在则创建
            os.makedirs(directory)
        
        directory = f"{self.home}/txt"
        # 检测目录是否存在
        if not os.path.exists(directory):
            # 如果不存在则创建
            os.makedirs(directory)
        
        
        # 保存JSON
        json_path = f"{self.home}/json/{base_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "meta": {
                    "created_at": datetime.now().isoformat(),
                    "model": "deepseek-chat",
                    "message_count": len(self.messages) - 1  # 排除系统消息
                },
                "messages": self.messages[1:]  # 排除系统消息
            }, f, ensure_ascii=False, indent=2)

        # 保存TXT
        txt_path = f"{self.home}/txt/{base_name}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"DeepSeek 对话记录 ({timestamp})\n")
            f.write("="*50 + "\n")
            for idx, msg in enumerate(self.messages[1:]):
                role = "用户" if msg['role'] == 'user' else "助手"
                f.write(f"[{idx//2+1}] {role}：{msg['content']}\n")

        return True


    def _generate_title(self):
        """使用对话内容生成标题"""
        try:
            # 提取前两轮对话作为生成依据
            sample_msgs = self.messages[1:3]  # 跳过系统消息，取前两个用户/助手消息
            if not sample_msgs:
                return None
                
            # 构建标题生成prompt
            title_prompt = [
                {
                    "role": "system",
                    "content": "你是一个专业的对话标题生成器，请根据以下对话内容生成一个简洁的中文标题，不超过15个字，不要使用任何标点符号。直接返回标题，不要解释。"
                },
                {
                    "role": "user",
                    "content": "\n".join([f"{msg['role']}: {msg['content']}" for msg in sample_msgs])
                }
            ]
            
            # 调用API生成标题
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=title_prompt,
                temperature=0.3,
                max_tokens=20
            )
            
            # 清理生成的标题
            raw_title = response.choices[0].message.content.strip()
            return self._sanitize_filename(raw_title)
            
        except Exception as e:
            print(f"标题生成失败：{str(e)}")
            return None

    @staticmethod
    def _sanitize_filename(title):
        """清理文件名中的非法字符"""
        # 保留允许的字符：中文、字母、数字、下划线和空格
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9_ ]', '', title)
        # 去除首尾空格并截断到30字符
        return cleaned.strip()[:30]




def get_user_input(prompt):
    readline.add_history(prompt)
    user_input = input(prompt)
    return user_input


def main():
    try:
        chat = DeepSeekChat()
        print("DeepSeek (V3)\n")
        
        while True:
            try:
                user_input = get_user_input("user:")
                if user_input.lower() in ('exit', 'quit'):
                    break
                if not user_input.strip():
                    continue
                
                chat.add_message("user", user_input)
                chat.stream_chat()
            
            except KeyboardInterrupt:
                print("\nInterrupt detected,exit.")
                break
        
        print("\nfinal conversation history：")
        for idx, msg in enumerate(chat.messages[1:]):  # 跳过系统消息
            print(f"[{idx//2+1}] {msg['role'].capitalize()}: {msg['content']}")
        chat.save_conversation()
    
    except Exception as e:
        print(f"program init failure.....：{str(e)}")
    finally:
        if hasattr(chat, 'sanitizer') and chat.sanitizer.error_log:
            print(f"\ndetected {len(chat.sanitizer.error_log)} abnormal characters，see char_errors.log")

if __name__ == "__main__":
    main()
