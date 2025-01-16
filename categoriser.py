import ollama
import os

model = "llama3.2"

#paths to input and output files
input_file = "./data/grocery_list.txt"
output_file = "./data/categorized_grocery_list.txt"

if not os.path.exists(input_file):
    print(f"Input file '{input_file}' not found.")
    exit(1)

#Opens the file . with securely closes the file
#open the file in read mode "r" 
# Stores file object in a variable called f
# f.strip() whitespace
#f.split() splits at whitespace
with open(input_file, "r") as f:
    items = f.read().strip()

#Prepare the promt for the model 
#f allows you to insert the variables in {} 
prompt = f"""
You are an assistant that categorizes and sorts grocery items

Here is a list of grocery items

{items}

Please:

1. Categorize these items into appropriate categories such as Produce, Dairy, Meat, Bakery, Beverages, etc.
2. Sort the items alphabetically within each category.
3. Present the categorized list in a clear and organized manner, using bullet points or numbering.




"""

# Attempt to generate a response from the Ollama model with streaming enabled.
# The 'stream=True' parameter tells the generate method to return the response in chunks as they are generated.
#response = ollama.generate(model=model, prompt=prompt, stream=True)

# Print a header to indicate the start of the categorized list.
#print("========Categorized list=========")

# Iterate over each chunk of the streamed response.
# The response is assumed to be a generator or iterable, providing data in parts.
#for chunk in response:
    # Extract the text part of the chunk using the "response" key.
    # If the "response" key is not found, use an empty string as a fallback.
    # 'chunk.get("response", "")' retrieves the incremental text.
    # 'end=""' prevents a newline after each chunk, ensuring the output flows smoothly.
    # 'flush=True' forces the output to display immediately without buffering.
 #   print(chunk.get("response", ""), end="", flush=True)


try:
    # Attempt to generate a response from the Ollama model
    response = ollama.generate(model=model , prompt=prompt)

    generated_text = response.get("response", " ")
    print("========Categorized list========= ")
    print(generated_text)

    # Open the output file in write mode, ensuring it gets closed after writing
    with open(output_file, "w") as f:
        f.write(generated_text.strip())

    # Inform the user that the categorized grocery list has been saved
    print(f"Categorized grocery list has been saved to '{output_file}' ")

except Exception as e:
    # Catch any exception that occurs and print the error message
    # str(e) converts the exception object into a human-readable string that usually contains the error message or details.
    print("An error occurred: ", str(e))    