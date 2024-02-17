# Chatbot interface
def chatbot_interface():
    while True:
        user_input = input("Enter a topic for a joke (type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        else:
            joke = generate_joke(fine_tuned_model, user_input)
            print("Here's a joke for you:")
            print(joke)

# Function to generate a joke based on the given topic
def generate_joke(model, topic):
    # Prompt the model with the topic and let it generate a joke
    prompt = f"Tell a joke about {topic}:"
    joke = model.generate(prompt)
    return joke

# Main function
if __name__ == "__main__":
    chatbot_interface()