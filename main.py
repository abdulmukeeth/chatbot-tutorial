import warnings
from langchain import PromptTemplate, LLMChain
from gemini_llm import GeminiLLM

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    # Initialize the custom Gemini LLM
    gemini_llm = GeminiLLM()

    # Define a prompt template
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="Answer the following question:\n\n{question}"
    )

    # Create an LLM Chain
    llm_chain = LLMChain(llm=gemini_llm, prompt=prompt_template)

    print("Welcome to the Gemini Interactive Chat!")
    print("Type your questions below. Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            # Check for exit commands
            if user_input.lower() in ['exit', 'quit']:
                print("Assistant: Goodbye! Have a great day.")
                break

            # Skip empty inputs
            if not user_input:
                print("Assistant: Please enter a valid question.")
                continue

            # Generate the answer
            answer = llm_chain.run(user_input)

            print(f"Assistant: {answer}\n")

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nAssistant: Session terminated. Goodbye!")
            break
        except Exception as e:
            # Handle unexpected errors
            print(f"Assistant: An error occurred: {e}\n")

if __name__ == "__main__":
    main()
