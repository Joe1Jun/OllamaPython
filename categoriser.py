import ollama
import os

model = "llam3.2"

#paths to input and output files
input_file = "./data/grocery_list.txt"
output_file = "./data/categorized_grocery_list.txt"

if not os.path.exists(input_file):
    print(f"Input file '{input_file}' not found.")
    exit(1)

with open(input_file, "r") as f:
    items = f.read().strip()