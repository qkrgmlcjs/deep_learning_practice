class Man:  #class 선언 키워드는 소문자
    def __init__(self, name) :
        self.name = name
        print("Initialized!")
        
    def hello(self) :
            print("Hello " + self.name + "!")
            
    def goodbye(self) :
        print("Good-bye " + self.name +"!")
        
m = Man("David")
m.hello()
m.goodbye()