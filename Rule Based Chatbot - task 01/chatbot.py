import datetime

def chatbot():
    print("Chatbot: Hello! How can I help you today?")
    
    while True:
        user_input = input("You: ").lower()
        
        if "hello" in user_input or "hi" in user_input:
            print("Chatbot: Hello! How are you?")
        
        elif "how are you" in user_input:
            print("Chatbot: I'm just a program, so I don't have feelings, but thanks for asking! How about you?")
        
        elif "your name" in user_input:
            print("Chatbot: I'm a simple chatbot created to help you with basic queries.")
        
        elif "time" in user_input:
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            print(f"Chatbot: The current time is {current_time}.")
        
        elif "weather" in user_input:
            print("Chatbot: I can't check the weather for you, but you can use a weather app or website for the latest updates!")
        
        elif "advice" in user_input:
            print("Chatbot: Remember to stay positive and keep learning new things every day!")
        
        elif "bye" in user_input or "goodbye" in user_input:
            print("Chatbot: Goodbye! Have a nice day!")
            break
        
        elif "help" in user_input or "support" in user_input:
            print("Chatbot: I'm here to assist you. You can ask me simple questions or type 'bye' to end the conversation.")
        
        else:
            print("Chatbot: I'm sorry, I don't understand that. Could you please rephrase?")
        
if __name__ == "__main__":
    chatbot()
