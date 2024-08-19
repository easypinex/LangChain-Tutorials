from langchain.runnables import Runnable

class MyCustomRunnable(Runnable):
    def __init__(self, custom_param):
        # 在初始化时传入自定义参数
        self.custom_param = custom_param
    
    def invoke(self, input_data):
        # 这里实现你自定义的逻辑
        print(f"Custom logic with param: {self.custom_param}")
        # 例如，简单返回输入数据和自定义参数
        return f"Processed {input_data} with {self.custom_param}"